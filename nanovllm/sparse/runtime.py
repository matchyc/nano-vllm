from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import torch

from .ann_index import ANNIndex
from .ops import sparse_subset_attention

logger = logging.getLogger("nanovllm.sparse.runtime")
_WARNED_MESSAGES: set[str] = set()


@dataclass
class LayerSequenceState:
    seq_id: int
    expected_len: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    metric: str
    ann_mode: str
    ivf_num_lists: int
    ivf_num_probe: int
    ivf_max_iters: int
    topk: int
    key_chunks: list[np.ndarray] = field(default_factory=list)
    slot_chunks: list[torch.Tensor] = field(default_factory=list)
    prefill_slots: torch.Tensor | None = None
    index: ANNIndex | None = None
    collected_len: int = 0
    prefill_len: int = 0
    build_latency_ms: float = 0.0

    def append(self, keys: torch.Tensor, slots: torch.Tensor):
        """Store a copy of prefills on CPU for later indexing."""
        if keys.numel() == 0:
            return
        keys_cpu = keys.to(torch.float32, copy=True).reshape(keys.size(0), -1).cpu().numpy()
        slots_cpu = slots.to(torch.int32, copy=True).cpu()
        self.key_chunks.append(keys_cpu)
        self.slot_chunks.append(slots_cpu)
        self.collected_len += keys.size(0)

    def build_index(self, metric: str):
        if self.index is not None or self.collected_len == 0:
            return False
        if self.collected_len < self.expected_len:
            return False
        corpus = np.concatenate(self.key_chunks, axis=0)
        self.prefill_slots = torch.cat(self.slot_chunks, dim=0)
        self.prefill_len = int(self.prefill_slots.numel())
        index = ANNIndex(
            metric=metric,
            mode=self.ann_mode,
            ivf_num_lists=self.ivf_num_lists,
            ivf_num_probe=self.ivf_num_probe,
            ivf_max_iters=self.ivf_max_iters,
        )
        t0 = perf_counter()
        index.build(corpus)
        self.build_latency_ms = (perf_counter() - t0) * 1000
        self.index = index
        self.key_chunks.clear()
        self.slot_chunks.clear()
        logger.info("Built ANN index: seq=%d tokens=%d latency=%.2fms", self.seq_id, self.prefill_len, self.build_latency_ms)
        return True

    def query(self, q_vector: torch.Tensor, k: int) -> torch.Tensor:
        assert self.index is not None and self.prefill_slots is not None
        if self.prefill_len == 0:
            raise RuntimeError("Index built without prefill tokens")
        k = min(k, self.prefill_len)
        q_np = q_vector.reshape(1, -1).to(torch.float32).cpu().numpy()
        idx = self.index.query(q_np, k)
        return torch.from_numpy(idx[0]).to(torch.int64)


class SparseAttentionManager:

    def __init__(self, config, hf_config):
        self.config = config
        self.enabled = bool(config.use_sparse_attention)
        self.block_size = config.kvcache_block_size
        self.metric = config.sparse_distance_metric
        self.topk = config.sparse_topk
        self.min_seq_len = config.sparse_min_seq_len
        self.granularity = config.sparse_index_granularity
        self.decode_window = config.sparse_decode_dense_window
        self.ann_mode = config.sparse_ann_mode
        self.ivf_num_lists = config.sparse_ivf_num_lists
        self.ivf_num_probe = config.sparse_ivf_num_probe
        self.ivf_max_iters = config.sparse_ivf_max_iters
        self.num_layers = hf_config.num_hidden_layers
        self.layer_states: List[Dict[int, LayerSequenceState]] = [dict() for _ in range(self.num_layers)]
        self.layer_meta: Dict[int, dict] = {}
        self.runtime_active = self.enabled

    def set_runtime_active(self, active: bool):
        self.runtime_active = active

    def attach_model(self, model):
        if not self.enabled:
            return
        if not hasattr(model, "model"):
            return
        layers = getattr(model, "model").layers
        for layer_id, layer in enumerate(layers):
            attn = layer.self_attn.attn
            attn.attach_sparse_manager(self, layer_id)
            self.layer_meta[layer_id] = dict(
                num_heads=attn.num_heads,
                num_kv_heads=attn.num_kv_heads,
                head_dim=attn.head_dim,
                scale=attn.scale,
            )

    # Prefill -----------------------------------------------------------------
    def on_prefill(self, layer_id: int, context, key_tensor: torch.Tensor):
        if not self._prefill_enabled(context):
            return
        slot_mapping = context.slot_mapping
        seq_ids = context.seq_ids or []
        cu_q = context.cu_seqlens_q
        cu_k = context.cu_seqlens_k or context.cu_seqlens_q
        if slot_mapping is None or cu_q is None or cu_k is None:
            return
        slot_cpu = slot_mapping.to("cpu")
        cu_q_list = cu_q.to("cpu").tolist()
        cu_k_list = cu_k.to("cpu").tolist()
        for idx, seq_id in enumerate(seq_ids):
            start = cu_q_list[idx]
            end = cu_q_list[idx + 1]
            expected_len = cu_k_list[idx + 1] - cu_k_list[idx]
            if end <= start or expected_len <= 0:
                continue
            seq_keys = key_tensor[start:end]
            if seq_keys.numel() == 0:
                continue
            state = self._get_or_create_state(layer_id, seq_id, expected_len)
            state.append(seq_keys, slot_cpu[start:end])
            if state.collected_len >= state.expected_len and state.index is None:
                state.build_index(self.metric)

    def _prefill_enabled(self, context) -> bool:
        if not self.enabled or not self.runtime_active:
            return False
        if context.seq_ids is None:
            return False
        if self.granularity != "layer_shared":
            msg = "sparse_index_granularity=per_head not implemented; falling back to dense."
            if msg not in _WARNED_MESSAGES:
                logger.warning(msg)
                _WARNED_MESSAGES.add(msg)
            return False
        return True

    def _get_or_create_state(self, layer_id: int, seq_id: int, expected_len: int) -> LayerSequenceState:
        layer_state = self.layer_states[layer_id]
        state = layer_state.get(seq_id)
        if state is None:
            meta = self.layer_meta[layer_id]
            state = LayerSequenceState(
                seq_id=seq_id,
                expected_len=expected_len,
                num_heads=meta["num_heads"],
                num_kv_heads=meta["num_kv_heads"],
                head_dim=meta["head_dim"],
                metric=self.metric,
                ann_mode=self.ann_mode,
                ivf_num_lists=self.ivf_num_lists,
                ivf_num_probe=self.ivf_num_probe,
                ivf_max_iters=self.ivf_max_iters,
                topk=self.topk,
            )
            layer_state[seq_id] = state
        else:
            state.expected_len = max(state.expected_len, expected_len)
        return state

    # Decode ------------------------------------------------------------------
    def try_decode(self, layer_id: int, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, context):
        if not self._can_use_sparse(layer_id, context):
            return None
        device = q.device
        k_flat = k_cache.view(-1, self.layer_meta[layer_id]["num_kv_heads"], self.layer_meta[layer_id]["head_dim"])
        v_flat = v_cache.view_as(k_flat)
        outputs = []
        seq_ids = context.seq_ids or []
        context_lens = context.context_lens.to("cpu").tolist()
        block_tables = context.block_tables
        for row, seq_id in enumerate(seq_ids):
            state = self.layer_states[layer_id][seq_id]
            q_vec = q[row]
            neighbors = state.query(q_vec.reshape(-1), self.topk)
            if neighbors.numel() == 0:
                return None
            indices = neighbors.to(torch.int64)
            subset_slots = state.prefill_slots.index_select(0, indices.to(torch.int64))
            slot_gpu = subset_slots.to(device=device, dtype=torch.long)
            prefill_k = k_flat.index_select(0, slot_gpu)
            prefill_v = v_flat.index_select(0, slot_gpu)
            decode_slots = self._recent_decode_slots(block_tables[row] if block_tables is not None else None,
                                                     context_lens[row], state.prefill_len, device)
            if decode_slots is not None:
                decode_k = k_flat.index_select(0, decode_slots)
                decode_v = v_flat.index_select(0, decode_slots)
                k_subset = torch.cat([prefill_k, decode_k], dim=0)
                v_subset = torch.cat([prefill_v, decode_v], dim=0)
            else:
                k_subset = prefill_k
                v_subset = prefill_v
            output = sparse_subset_attention(q_vec, k_subset, v_subset,
                                             state.num_heads, state.num_kv_heads,
                                             self.layer_meta[layer_id]["scale"])
            outputs.append(output)
        return torch.stack(outputs, dim=0)

    def _can_use_sparse(self, layer_id: int, context) -> bool:
        if not self.enabled or not self.runtime_active:
            return False
        if context.seq_ids is None or context.context_lens is None:
            return False
        if self.granularity != "layer_shared":
            return False
        seq_ids = context.seq_ids
        context_lens = context.context_lens.to("cpu").tolist()
        for idx, seq_id in enumerate(seq_ids):
            state = self.layer_states[layer_id].get(seq_id)
            if state is None or state.index is None or state.prefill_len == 0:
                return False
            if context_lens[idx] < self.min_seq_len:
                return False
        return True

    def _recent_decode_slots(self, block_row: torch.Tensor | None, total_len: int, prefill_len: int, device) -> Optional[torch.Tensor]:
        if block_row is None:
            return None
        decode_len = max(total_len - prefill_len, 0)
        if decode_len == 0 or self.decode_window == 0:
            return None
        window = min(self.decode_window, decode_len)
        start = total_len - window
        positions = torch.arange(start, total_len, dtype=torch.int64, device="cpu")
        num_blocks = math.ceil(total_len / self.block_size)
        block_ids = block_row.to("cpu")[:num_blocks].to(torch.int64)
        block_idx = torch.div(positions, self.block_size, rounding_mode="floor")
        offsets = torch.remainder(positions, self.block_size)
        slots = block_ids.index_select(0, block_idx) * self.block_size + offsets
        return slots.to(device=device, dtype=torch.long)
