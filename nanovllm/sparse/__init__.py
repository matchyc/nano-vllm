__all__ = ["SparseAttentionManager", "ANNIndex"]


def __getattr__(name):
    if name == "SparseAttentionManager":
        from .runtime import SparseAttentionManager as _SparseAttentionManager
        return _SparseAttentionManager
    if name == "ANNIndex":
        from .ann_index import ANNIndex as _ANNIndex
        return _ANNIndex
    raise AttributeError(name)
