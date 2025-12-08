from __future__ import annotations

import torch


def _expand_for_heads(kv_tokens: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """
    kv_tokens: [num_tokens, num_kv_heads, head_dim]
    Returns: [num_heads, num_tokens, head_dim]
    """
    if num_heads == num_kv_heads:
        expanded = kv_tokens
    else:
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        repeat = num_heads // num_kv_heads
        expanded = kv_tokens.repeat_interleave(repeat, dim=1)
    return expanded.permute(1, 0, 2)


def scaled_dot_product(q: torch.Tensor, k_tokens: torch.Tensor, v_tokens: torch.Tensor, scale: float) -> torch.Tensor:
    """
    q: [num_heads, head_dim]
    k_tokens: [num_heads, num_tokens, head_dim]
    v_tokens: [num_heads, num_tokens, head_dim]
    """
    q32 = q.to(torch.float32)
    k32 = k_tokens.to(torch.float32)
    v32 = v_tokens.to(torch.float32)
    scores = torch.matmul(q32.unsqueeze(1), k32.transpose(-1, -2)).squeeze(1)
    scores = scores * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs.unsqueeze(1), v32).squeeze(1)
    return out.to(q.dtype)


def dense_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                              num_heads: int, num_kv_heads: int, scale: float) -> torch.Tensor:
    """
    Reference dense attention used in tests.
    q: [num_heads, head_dim]
    k/v: [num_tokens, num_kv_heads, head_dim]
    """
    k_heads = _expand_for_heads(k, num_heads, num_kv_heads)
    v_heads = _expand_for_heads(v, num_heads, num_kv_heads)
    return scaled_dot_product(q, k_heads, v_heads, scale)


def sparse_subset_attention(q: torch.Tensor, k_subset: torch.Tensor, v_subset: torch.Tensor,
                            num_heads: int, num_kv_heads: int, scale: float) -> torch.Tensor:
    """
    Computes attention over a subset of tokens. Used both by runtime and tests.
    k_subset/v_subset: [num_subset_tokens, num_kv_heads, head_dim]
    """
    k_heads = _expand_for_heads(k_subset, num_heads, num_kv_heads)
    v_heads = _expand_for_heads(v_subset, num_heads, num_kv_heads)
    return scaled_dot_product(q, k_heads, v_heads, scale)
