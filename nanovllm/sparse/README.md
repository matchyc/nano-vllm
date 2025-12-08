# Sparse Attention Module for nano-vllm (MLANN-based)

This module implements a prototype sparse attention mechanism for nano-vllm using the **MLANN algorithm** (Multilabel Classification for Approximate Nearest Neighbor Search) from NeurIPS 2022 / JMLR 2024.

## MLANN Algorithm Overview

MLANN treats ANN candidate selection as a **multilabel classification problem**:

1. **Training Phase** (Attention-Aware):
   - **Corpus vectors**: `{K_j}` (key vectors from prefill)
   - **Training queries**: `{Q_i}` (query vectors from prefill) - **NOT K!**
   - **Labels**: `Y_i = {indices of k-NN of Q_i in K space}`
   - Build a partitioning structure (RP tree ensemble) that divides the space into cells
   
   **Why Q as training queries?** In attention, Q queries K. The MLANN classifier should learn the Q→K attention pattern, not K→K self-similarity.

2. **Natural Classifier**:
   - For each training query `Q_i`, find its partition cell `r(Q_i)`
   - For each cell `r` and corpus index `j`, estimate:
     ```
     p(r, j) = P(K_j in k-NN of Q | Q lands in cell r)
             ≈ (# Q in cell r with K_j in their k-NN) / (# Q in cell r)
     ```

3. **Query Phase** (Decode):
   - Route new query `Q` to cell `r(Q)` via the same partitioning scheme
   - Score corpus indices by aggregating `p(r_t(Q), j)` across all trees `t`
   - Return top-k indices by score

## Key Components

### MLANNIndex (`mlann_index.py`)

Self-contained MLANN implementation with:

- **RP Tree Partitioner**: Random projection tree for space partitioning
- **Multi-tree Ensemble**: Multiple independent RP trees (like random forest) for better recall
- **Label Probability Estimation**: Per-cell probability tables for the natural classifier

```python
from nanovllm.sparse.mlann_index import MLANNIndex

# Create MLANN index
index = MLANNIndex(
    metric="ip",       # "ip" (inner product) or "l2"
    k_train=32,        # k for computing training labels (k-NN ground truth)
    n_trees=8,         # Number of RP trees in ensemble
    max_depth=8,       # Maximum tree depth
    min_leaf_size=10,  # Minimum points per leaf
)

# Build index (uses corpus as both corpus and training queries)
index.build(corpus)  # corpus: [num_tokens, dim]

# Query for top-k candidates
indices = index.query(queries, k=64)  # returns [num_queries, k]
```

### SparseAttentionManager (`manager.py`)

Manages per-layer MLANN indices:

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
    sparse_index_granularity="layer_shared",
    sparse_include_decode_dense=True,
    sparse_max_decode_tokens=64,
    sparse_debug=False,
    
    # MLANN-specific parameters
    sparse_mlann_k_train=32,        # Training k-NN size
    sparse_mlann_n_trees=8,         # Number of RP trees
    sparse_mlann_max_depth=8,       # Max tree depth
    sparse_mlann_min_leaf_size=10,  # Min leaf size
)
```

## Algorithm Details

### Random Projection Tree (RP Tree)

Each tree is built by:
1. Choose a random projection direction (Gaussian)
2. Compute projection values for all points
3. Split at median (balanced partition)
4. Recurse until max_depth or min_leaf_size reached

### Training Label Computation

For each query vector `Q_i`:
1. Compute exact k-NN in K (corpus) using brute-force: find which K vectors are most similar to Q_i
2. Store indices of k nearest K neighbors as labels `Y_i`

This captures the actual attention pattern: which K tokens does each Q attend to most strongly?

### Per-Cell Probability Estimation

For each (cell, corpus_index) pair:
```
p(cell, j) = count(queries in cell with j in k-NN) / count(queries in cell)
```

### Multi-Tree Score Aggregation

At query time:
```
score(j) = Σ_t p(r_t(q), j)  # Sum across all trees
```

Return top-k by aggregated score.

## Expected Recall

MLANN achieves best recall when queries come from the same distribution as training data:

| Configuration | Same-Distribution Recall@16 |
|--------------|----------------------------|
| 1 tree, depth=8 | ~10-15% |
| 8 trees, depth=8 | ~40-50% |
| 16 trees, depth=8 | ~50-60% |

For sparse attention, Q and K come from the same transformer layer, so they share similar distributions.

## Limitations (v1)

1. **Single Sequence**: Sparse attention for batch size 1 only
2. **No Online Updates**: Index only covers prefill tokens
3. **Synchronous Build**: Index building is synchronous (designed for async extension)
4. **CPU Index**: Index operations run on CPU (could be GPU-accelerated)

## Testing

Run unit tests:
```bash
python tests/test_sparse_attention.py
```

Run MLANN standalone test:
```bash
python nanovllm/sparse/mlann_index.py
```

## References

- Paper: "A Multilabel Classification Framework for Approximate Nearest Neighbor Search"
  - NeurIPS 2022 / JMLR 2024
  - https://www.jmlr.org/papers/volume25/23-0286/23-0286.pdf

## Implementation Notes

This implementation follows the MLANN natural classifier formulation:
- Uses RP tree partitioning (paper also discusses PCA-based and classifier-based options)
- Multi-tree ensemble for improved recall (similar to random forest idea in paper)
- Pure Python/NumPy implementation, no external ANN libraries
