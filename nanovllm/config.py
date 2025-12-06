import os
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
    # Sparse attention configuration
    use_sparse_attention: bool = False
    sparse_topk: int = 64
    sparse_min_seq_len: int = 512
    sparse_distance_metric: str = "ip"  # "ip" or "l2"
    sparse_index_granularity: str = "layer_shared"  # "layer_shared" or "per_head"
    # MLANN build parameters
    sparse_mlann_n_trees: int = 10
    sparse_mlann_depth: int = 6
    sparse_mlann_votes_required: int = 5

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
