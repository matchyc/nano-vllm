# Sparse Attention Module for nano-vllm

This module implements a prototype sparse attention mechanism for nano-vllm, enabling efficient attention computation over long contexts by attending only to the most relevant keys.

## Overview

The sparse attention system works in two phases:

1. **Prefill Phase**: After computing K/V for all prompt tokens, we build an ANN (Approximate Nearest Neighbor) index over the key vectors.

2. **Decode Phase**: Instead of dense attention over all past keys, we:
   - Query the ANN index to find the top-k most relevant keys for each new query
   - Compute attention only over this sparse subset
   - Optionally include dense attention over recent decode tokens

## Key Components

### ANNIndex (`ann_index.py`)

Self-contained ANN index with two backends:

- **Exact Mode** (`mode="exact"`): Brute-force kNN search. O(n) per query but provides exact results. Good baseline for correctness testing.

- **IVF Mode** (`mode="ivf"`): Inverted File index using k-means clustering. Faster query time but approximate results. Configurable recall/speed tradeoff via `nprobe`.

```python
from nanovllm.sparse.ann_index import ANNIndex

# Create index
index = ANNIndex(metric="ip", mode="exact")  # or mode="ivf"

# Build with corpus vectors [num_tokens, dim]
index.build(corpus)

# Query with query vectors [num_queries, dim]
indices = index.query(queries, k=64)  # returns [num_queries, k]
```

### SparseAttentionManager (`manager.py`)

Manages per-layer indices and coordinates sparse attention:

- Extracts keys from KV cache after prefill
- Builds indices (synchronously in v1, designed for async extension)
- Handles queries during decode
- Supports two granularity modes:
  - `layer_shared`: One index per layer, all heads share
  - `per_head`: One index per head (partially implemented)

```python
from nanovllm.sparse.manager import SparseAttentionManager

manager = SparseAttentionManager(config, num_layers, num_kv_heads, head_dim)
manager.build_indices_from_kv_cache(kv_cache, seq_lens, block_tables, block_size)
indices = manager.query(layer_id, queries, k=64)
```

## Configuration

Enable sparse attention via Config:

```python
from nanovllm import LLM

llm = LLM(
    model_path,
    use_sparse_attention=True,      # Master switch
    sparse_topk=64,                 # Number of keys to attend to
    sparse_min_seq_len=512,         # Min seq len to enable sparse
    sparse_distance_metric="ip",    # "ip" or "l2"
    sparse_index_granularity="layer_shared",  # or "per_head"
    sparse_ann_mode="exact",        # "exact" or "ivf"
    sparse_ivf_nlist=64,            # IVF clusters
    sparse_ivf_nprobe=8,            # IVF probes
    sparse_include_decode_dense=True,  # Include decode tokens
    sparse_max_decode_tokens=64,    # Max decode tokens to include
    sparse_debug=False,             # Debug logging
)
```

## Fallback Behavior

When `use_sparse_attention=False` (default), behavior is **identical** to original nano-vllm.

When enabled:
- Sparse attention only activates when `seq_len >= sparse_min_seq_len`
- Otherwise falls back to dense attention
- Missing indices also trigger fallback

## Limitations (v1)

1. **Single Sequence**: Currently only supports batch size 1 for sparse attention
2. **No Online Updates**: ANN index only covers prefill tokens; decode tokens are not added
3. **Synchronous Build**: Index building is synchronous (designed for async extension)
4. **Single GPU**: No multi-GPU optimizations for sparse path

## Performance Characteristics

### Index Build Time (per layer)
- Exact: O(n) - just stores vectors
- IVF: O(n * k-means iterations) - needs clustering

### Query Time (per query batch)
- Exact: O(n * d) - full similarity computation
- IVF: O(nprobe * cluster_size * d) - subset search

### Memory
- CPU memory for indices (corpus stored as float32)
- GPU memory unchanged (KV cache remains on GPU)

## Future Improvements (TODO)

- [ ] Batched sparse attention support
- [ ] Async/overlapped index building
- [ ] Online index updates during decode
- [ ] Custom CUDA kernels for sparse gather
- [ ] More ANN backends (HNSW, LSH)
- [ ] Per-head index optimization

## Testing

Run unit tests:
```bash
python tests/test_sparse_attention.py
```

Run benchmarks:
```bash
python bench_sparse.py
```

## Design Rationale

The design is inspired by MLANN but implemented as a self-contained module:

1. **No External Dependencies**: Only uses NumPy/PyTorch, no FAISS/HNSWlib
2. **Minimal Changes**: Core nano-vllm code changes are small and reversible
3. **Clear Separation**: Sparse logic isolated in `sparse/` module
4. **Extensible**: API designed for future optimizations (async, batching, etc.)
