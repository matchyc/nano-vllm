from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import torch


@dataclass
class SparseAttentionContext:
    """Context for sparse attention computation during decode."""
    
    # Whether sparse attention is enabled for this step
    enabled: bool = False
    
    # Layer-specific sparse indices: layer_id -> [batch_size, topk] indices into prefill tokens
    sparse_indices: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    # Decode token range to include with dense attention (start, end)
    # If None, only sparse attention over prefill tokens
    decode_range: Optional[Tuple[int, int]] = None
    
    # Prefill sequence length (tokens covered by the ANN index)
    prefill_len: int = 0
    
    # Current total sequence length
    current_seq_len: int = 0
    
    # Reference to sparse attention manager (set during decode)
    manager: Any = None  # SparseAttentionManager


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    
    # Sparse attention context (only used during decode when sparse attention is enabled)
    sparse: Optional[SparseAttentionContext] = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill, 
    cu_seqlens_q=None, 
    cu_seqlens_k=None, 
    max_seqlen_q=0, 
    max_seqlen_k=0, 
    slot_mapping=None, 
    context_lens=None, 
    block_tables=None,
    sparse=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill, cu_seqlens_q, cu_seqlens_k, 
        max_seqlen_q, max_seqlen_k, 
        slot_mapping, context_lens, block_tables,
        sparse,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
