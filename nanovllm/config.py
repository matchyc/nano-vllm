import os
from dataclasses import dataclass, field
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
    
    # ============ Sparse Attention Configuration ============
    # Master switch for sparse attention (default: disabled for identical behavior)
    use_sparse_attention: bool = False
    # Number of top keys to attend to in sparse mode
    sparse_topk: int = 64
    # Minimum sequence length to enable sparse attention (fallback to dense below this)
    sparse_min_seq_len: int = 512
    # Distance metric for ANN: "ip" (inner product) or "l2" (Euclidean distance)
    sparse_distance_metric: str = "ip"
    # Index granularity: "layer_shared" (one index per layer, all heads share)
    #                    "per_head" (one index per head) - partially implemented
    sparse_index_granularity: str = "layer_shared"
    # ANN mode: "exact" (brute-force kNN) or "ivf" (approximate IVF-style)
    sparse_ann_mode: str = "exact"
    # Number of clusters for IVF mode (only used when sparse_ann_mode="ivf")
    sparse_ivf_nlist: int = 64
    # Number of clusters to probe in IVF mode
    sparse_ivf_nprobe: int = 8
    # Whether to include decode tokens in attention (dense) alongside sparse prefill
    # If True: sparse over prefill + dense over recent decode tokens
    # If False: sparse over prefill only (simpler, use for short decode)
    sparse_include_decode_dense: bool = True
    # Maximum number of recent decode tokens to include in dense attention
    sparse_max_decode_tokens: int = 64
    # Enable debug logging for sparse attention
    sparse_debug: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        
        # Validate sparse attention config
        if self.use_sparse_attention:
            assert self.sparse_topk > 0, "sparse_topk must be positive"
            assert self.sparse_min_seq_len > 0, "sparse_min_seq_len must be positive"
            assert self.sparse_distance_metric in ("ip", "l2"), \
                "sparse_distance_metric must be 'ip' or 'l2'"
            assert self.sparse_index_granularity in ("layer_shared", "per_head"), \
                "sparse_index_granularity must be 'layer_shared' or 'per_head'"
            assert self.sparse_ann_mode in ("exact", "ivf"), \
                "sparse_ann_mode must be 'exact' or 'ivf'"
            if self.sparse_ann_mode == "ivf":
                assert self.sparse_ivf_nlist > 0, "sparse_ivf_nlist must be positive"
                assert self.sparse_ivf_nprobe > 0, "sparse_ivf_nprobe must be positive"
