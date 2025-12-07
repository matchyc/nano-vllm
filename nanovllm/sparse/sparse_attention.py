"""
Sparse attention computation utilities.

This module provides functions for computing sparse attention using ANN indices.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def sparse_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Compute sparse attention during prefill (currently falls back to dense).
    
    In v1, prefill always uses dense attention. This function is a placeholder
    for future sparse prefill support.
    
    Args:
        q: Query tensor [total_q_tokens, num_heads, head_dim]
        k: Key tensor [total_k_tokens, num_kv_heads, head_dim]
        v: Value tensor [total_v_tokens, num_kv_heads, head_dim]
        scale: Attention scale factor
        
    Returns:
        Output tensor [total_q_tokens, num_heads, head_dim]
    """
    # For v1, prefill always uses dense attention
    # This is a placeholder for future sparse prefill
    raise NotImplementedError("Sparse prefill not implemented in v1")


def sparse_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    block_table: List[int],
    block_size: int,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Compute sparse attention during decode using ANN indices.
    
    This function gathers K/V from the prefill region using ANN indices,
    then computes attention. For v1, decode-phase tokens are handled
    separately (either ignored or attended densely over a small window).
    
    Args:
        q: Query tensor [batch_size, num_heads, head_dim]
        k_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim]
        indices: ANN indices [batch_size, topk] pointing to prefill tokens (0-based within sequence)
        block_table: Block table mapping logical blocks to physical blocks
        block_size: Size of each block
        scale: Attention scale factor
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimension of each head
        
    Returns:
        Output tensor [batch_size, num_heads, head_dim]
    """
    batch_size = q.shape[0]
    topk = indices.shape[1]
    num_blocks, block_size_dim, _, _ = k_cache.shape
    
    # Map sequence indices to cache positions using block_table
    # indices: [batch_size, topk] are token positions within the sequence
    # We need to map them to (physical_block, offset) pairs
    k_gathered = torch.zeros(
        batch_size, topk, num_kv_heads, head_dim,
        dtype=k_cache.dtype,
        device=k_cache.device
    )
    v_gathered = torch.zeros(
        batch_size, topk, num_kv_heads, head_dim,
        dtype=v_cache.dtype,
        device=v_cache.device
    )
    
    for b in range(batch_size):
        for k_idx in range(topk):
            token_idx = indices[b, k_idx].item()
            logical_block = token_idx // block_size
            offset = token_idx % block_size
            
            if logical_block < len(block_table):
                physical_block = block_table[logical_block]
                if physical_block >= 0 and physical_block < num_blocks:
                    k_gathered[b, k_idx] = k_cache[physical_block, offset]
                    v_gathered[b, k_idx] = v_cache[physical_block, offset]
    
    # Expand q for multi-head: [batch_size, num_heads, 1, head_dim]
    q = q.unsqueeze(2)  # [batch_size, num_heads, 1, head_dim]
    
    # Expand k/v for GQA: [batch_size, 1, topk, num_kv_heads, head_dim]
    k_gathered = k_gathered.unsqueeze(1)  # [batch_size, 1, topk, num_kv_heads, head_dim]
    v_gathered = v_gathered.unsqueeze(1)  # [batch_size, 1, topk, num_kv_heads, head_dim]
    
    # Repeat k/v heads to match q heads if needed
    if num_heads != num_kv_heads:
        # GQA: repeat kv heads
        repeat_factor = num_heads // num_kv_heads
        k_gathered = k_gathered.repeat(1, repeat_factor, 1, 1, 1)
        v_gathered = v_gathered.repeat(1, repeat_factor, 1, 1, 1)
    
    # Reshape for matmul: [batch_size, num_heads, topk, head_dim]
    k_gathered = k_gathered.squeeze(1)  # [batch_size, num_heads, topk, head_dim]
    v_gathered = v_gathered.squeeze(1)  # [batch_size, num_heads, topk, head_dim]
    
    # Compute attention scores: [batch_size, num_heads, 1, topk]
    scores = torch.matmul(q, k_gathered.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, 1, topk]
    
    # Apply attention to values: [batch_size, num_heads, 1, head_dim]
    output = torch.matmul(attn_weights, v_gathered)  # [batch_size, num_heads, 1, head_dim]
    
    # Squeeze: [batch_size, num_heads, head_dim]
    output = output.squeeze(2)
    
    return output


def gather_kv_from_indices(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather K/V from cache using indices and block_table.
    
    This is a helper function that properly maps indices through the block_table
    to gather keys and values from the paged KV cache.
    
    Args:
        k_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim]
        indices: Token indices [batch_size, topk] (absolute positions in sequence)
        block_table: Block table [batch_size, max_blocks] mapping logical blocks to physical
        block_size: Size of each block
        num_kv_heads: Number of KV heads
        head_dim: Head dimension
        
    Returns:
        k_gathered: [batch_size, topk, num_kv_heads, head_dim]
        v_gathered: [batch_size, topk, num_kv_heads, head_dim]
    """
    batch_size, topk = indices.shape
    num_blocks, block_size_dim, _, _ = k_cache.shape
    
    # For each batch item and each index, compute block_idx and offset
    block_indices = indices // block_size  # [batch_size, topk]
    offsets = indices % block_size  # [batch_size, topk]
    
    # Map logical blocks to physical blocks using block_table
    # block_table: [batch_size, max_blocks]
    max_blocks = block_table.shape[1]
    
    k_gathered = torch.zeros(
        batch_size, topk, num_kv_heads, head_dim,
        dtype=k_cache.dtype,
        device=k_cache.device
    )
    v_gathered = torch.zeros(
        batch_size, topk, num_kv_heads, head_dim,
        dtype=v_cache.dtype,
        device=v_cache.device
    )
    
    for b in range(batch_size):
        for k_idx in range(topk):
            logical_block = block_indices[b, k_idx].item()
            offset = offsets[b, k_idx].item()
            
            if logical_block < max_blocks:
                physical_block = block_table[b, logical_block].item()
                if physical_block >= 0 and physical_block < num_blocks:
                    k_gathered[b, k_idx] = k_cache[physical_block, offset]
                    v_gathered[b, k_idx] = v_cache[physical_block, offset]
    
    return k_gathered, v_gathered
