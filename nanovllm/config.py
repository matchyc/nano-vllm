import os
import logging
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    use_sparse_attention: bool = False
    sparse_topk: int = 64
    sparse_min_seq_len: int = 512
    sparse_distance_metric: str = "ip"
    sparse_index_granularity: str = "layer_shared"
    sparse_decode_dense_window: int = 128
    sparse_index_num_trees: int = 4
    sparse_index_depth: int = 6

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        disable_sparse = os.getenv("NANOVLLM_DISABLE_SPARSE_ATTENTION", "").lower() in {"1", "true", "yes"}
        if disable_sparse:
            logging.warning("Sparse attention disabled via NANOVLLM_DISABLE_SPARSE_ATTENTION.")
            self.use_sparse_attention = False
        if self.use_sparse_attention:
            assert self.tensor_parallel_size == 1, "Sparse attention v1 only supports tensor_parallel_size == 1"
            assert self.sparse_index_granularity in {"layer_shared", "per_head"}
            assert self.sparse_distance_metric in {"ip", "l2"}
            assert self.sparse_topk > 0
            assert self.sparse_min_seq_len >= self.sparse_topk
            assert self.sparse_decode_dense_window >= 0
