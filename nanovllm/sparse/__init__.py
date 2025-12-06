__all__ = ["SparseAttentionManager"]


def __getattr__(name):
    if name == "SparseAttentionManager":
        from .runtime import SparseAttentionManager as _SparseAttentionManager
        return _SparseAttentionManager
    raise AttributeError(name)
