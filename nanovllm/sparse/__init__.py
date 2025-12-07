"""
Sparse Attention Module for nano-vllm

This module provides MLANN-based sparse attention for efficient long-context inference.
MLANN (Multilabel Classification for ANN) treats candidate selection as a multilabel
classification problem, using random projection tree ensembles for space partitioning.

The algorithm is reimplemented locally following the MLANN paper:
"A Multilabel Classification Framework for Approximate Nearest Neighbor Search"
(NeurIPS 2022 / JMLR 2024)

Key components:
- MLANNIndex: MLANN index with RP tree ensemble partitioner
- SparseAttentionManager: Manages per-layer index building and querying
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "MLANNIndex":
        from nanovllm.sparse.mlann_index import MLANNIndex
        return MLANNIndex
    elif name == "SparseAttentionManager":
        from nanovllm.sparse.manager import SparseAttentionManager
        return SparseAttentionManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MLANNIndex", "SparseAttentionManager"]
