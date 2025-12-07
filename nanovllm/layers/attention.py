import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.sparse.sparse_attention import sparse_attention_decode, gather_kv_from_indices
from typing import Optional


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        layer_id: Optional[int] = None,
        index_manager: Optional[object] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.layer_id = layer_id
        self.index_manager = index_manager

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        # Check if we should use sparse attention
        use_sparse = (
            self.index_manager is not None
            and self.layer_id is not None
            and self.index_manager.use_sparse
        )
        
        if context.is_prefill:
            # Prefill: always use dense attention in v1
            # (Sparse prefill can be added later)
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # Decode: check if sparse attention should be used
            if use_sparse and context.context_lens is not None:
                # Try sparse attention path
                batch_size = q.shape[0]
                seq_lens = context.context_lens.cpu().tolist()
                
                # For v1, we only support single sequence per batch for sparse attention
                # (Multi-sequence batching with sparse attention requires more complex handling)
                if batch_size == 1 and seq_lens[0] >= self.index_manager.min_seq_len:
                    # Query ANN index
                    # Reshape q: [batch_size, num_heads, head_dim] -> [batch_size, num_heads * head_dim]
                    q_flat = q.flatten(1)  # [batch_size, num_heads * head_dim]
                    
                    # For v1, we assume seq_id = 0 (single sequence)
                    # In a full implementation, we'd need to track seq_id per batch item
                    indices = self.index_manager.query_layer(
                        self.layer_id,
                        q_flat,
                        seq_id=0,  # TODO: get actual seq_id from context
                    )
                    
                    if indices is not None:
                        # Use sparse attention
                        # Get block_table for this sequence
                        block_table = self.index_manager.get_block_table(seq_id=0)
                        if block_table is not None:
                            try:
                                o = sparse_attention_decode(
                                    q,
                                    k_cache,
                                    v_cache,
                                    indices,
                                    block_table,
                                    self.index_manager.config.kvcache_block_size,
                                    self.scale,
                                    self.num_heads,
                                    self.num_kv_heads,
                                    self.head_dim,
                                )
                                # Sparse attention succeeded, return result
                                return o
                            except Exception as e:
                                # Fall back to dense if sparse fails
                                # In production, we'd log this
                                pass
            
            # Dense attention (fallback or default)
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
