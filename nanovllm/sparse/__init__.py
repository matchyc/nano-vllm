"""
Sparse Attention Module for nano-vllm

This module provides ANN-based sparse attention for efficient long-context inference.
The design is inspired by MLANN but implemented as a self-contained module using only
standard dependencies (NumPy, PyTorch).

Key components:
- ANNIndex: ANN index with exact and approximate (IVF) backends
- SparseAttentionManager: Manages per-layer index building and querying
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "ANNIndex":
        from nanovllm.sparse.ann_index import ANNIndex
        return ANNIndex
    elif name == "SparseAttentionManager":
        from nanovllm.sparse.manager import SparseAttentionManager
        return SparseAttentionManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ANNIndex", "SparseAttentionManager"]
