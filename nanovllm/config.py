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
    # Distance metric for MLANN: "ip" (inner product) or "l2" (Euclidean distance)
    sparse_distance_metric: str = "ip"
    # Index granularity: "layer_shared" (one index per layer, all heads share)
    #                    "per_head" (one index per head) - partially implemented
    sparse_index_granularity: str = "layer_shared"
    # Whether to include decode tokens in attention (dense) alongside sparse prefill
    # If True: sparse over prefill + dense over recent decode tokens
    # If False: sparse over prefill only (simpler, use for short decode)
    sparse_include_decode_dense: bool = True
    # Maximum number of recent decode tokens to include in dense attention
    sparse_max_decode_tokens: int = 64
    # Enable debug logging for sparse attention
    sparse_debug: bool = False
    
    # ============ MLANN Algorithm Configuration ============
    # MLANN uses multilabel classification for ANN candidate selection
    # (see "A Multilabel Classification Framework for Approximate Nearest Neighbor Search")
    
    # k_train: Number of neighbors for computing training labels (k-NN ground truth)
    # Larger k_train captures more potential neighbors but increases build cost
    sparse_mlann_k_train: int = 32
    # n_trees: Number of RP trees in the ensemble (random forest style)
    # More trees = better recall but slower build/query
    sparse_mlann_n_trees: int = 8
    # max_depth: Maximum depth of each RP tree
    # Deeper trees = more cells = finer partitioning
    sparse_mlann_max_depth: int = 8
    # min_leaf_size: Minimum points in a leaf node before stopping splits
    sparse_mlann_min_leaf_size: int = 10

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
            # MLANN config validation
            assert self.sparse_mlann_k_train > 0, "sparse_mlann_k_train must be positive"
            assert self.sparse_mlann_n_trees > 0, "sparse_mlann_n_trees must be positive"
            assert self.sparse_mlann_max_depth > 0, "sparse_mlann_max_depth must be positive"
            assert self.sparse_mlann_min_leaf_size > 0, "sparse_mlann_min_leaf_size must be positive"
