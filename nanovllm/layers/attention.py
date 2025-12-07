import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context, SparseAttentionContext


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# Global layer counter for sparse attention (reset each forward pass)
_SPARSE_LAYER_ID = 0

def reset_sparse_layer_counter():
    global _SPARSE_LAYER_ID
    _SPARSE_LAYER_ID = 0

def get_and_increment_sparse_layer_id() -> int:
    global _SPARSE_LAYER_ID
    layer_id = _SPARSE_LAYER_ID
    _SPARSE_LAYER_ID += 1
    return layer_id


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        
        # Layer ID will be set during forward for sparse attention
        self._layer_id: Optional[int] = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # Check if sparse attention should be used
            if context.sparse is not None and context.sparse.enabled:
                o = self._sparse_attention_decode(q, k_cache, v_cache, context)
            else:
                # Dense attention (original path)
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
        return o
    
    def _sparse_attention_decode(
        self, 
        q: torch.Tensor,  # [batch_size, num_heads, head_dim]
        k_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
        context,
    ) -> torch.Tensor:
        """
        Sparse attention for decode: attend to top-k prefill tokens + recent decode tokens.
        
        This implementation uses:
        1. ANN query to find top-k key indices from prefill region
        2. Gather K/V from cache for those indices
        3. Simple matmul-based softmax attention (no custom kernel needed for v1)
        
        Args:
            q: Query tensor [batch_size, num_heads, head_dim]
            k_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim]
            v_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim]
            context: Context with sparse attention info
            
        Returns:
            o: Output tensor [batch_size, num_heads, head_dim]
        """
        sparse_ctx = context.sparse
        manager = sparse_ctx.manager
        
        # Get layer ID (incremented each call during decode)
        layer_id = get_and_increment_sparse_layer_id()
        
        batch_size = q.shape[0]
        device = q.device
        dtype = q.dtype
        
        # Check if index exists for this layer
        if not manager.has_index(layer_id):
            # Fallback to dense attention
            return flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens, 
                block_table=context.block_tables,
                softmax_scale=self.scale, causal=True
            )
        
        # Query ANN index for top-k prefill token indices
        # q shape: [batch_size, num_heads, head_dim]
        sparse_indices = manager.query(layer_id, q, k=manager.topk)  # [batch_size, topk]
        
        # Gather K/V from cache for sparse indices
        # Need to map token indices to cache slots using block_table
        sparse_k, sparse_v = self._gather_kv_sparse(
            k_cache, v_cache, 
            sparse_indices, 
            context.block_tables,
            manager.config.kvcache_block_size,
        )  # [batch_size, topk, num_kv_heads, head_dim]
        
        # Optionally include recent decode tokens with dense attention
        if sparse_ctx.decode_range is not None:
            decode_start, decode_end = sparse_ctx.decode_range
            decode_k, decode_v = self._gather_kv_range(
                k_cache, v_cache,
                decode_start, decode_end,
                context.block_tables,
                manager.config.kvcache_block_size,
            )  # [batch_size, num_decode, num_kv_heads, head_dim]
            
            # Concatenate sparse prefill + dense decode
            sparse_k = torch.cat([sparse_k, decode_k], dim=1)  # [batch, topk+decode, heads, dim]
            sparse_v = torch.cat([sparse_v, decode_v], dim=1)
        
        # Compute attention with simple matmul (no custom kernel for v1)
        # Handle GQA: expand KV heads to match query heads
        num_kv_groups = self.num_heads // self.num_kv_heads
        if num_kv_groups > 1:
            # Expand KV for GQA: [batch, seq, num_kv_heads, dim] -> [batch, seq, num_heads, dim]
            sparse_k = sparse_k.repeat_interleave(num_kv_groups, dim=2)
            sparse_v = sparse_v.repeat_interleave(num_kv_groups, dim=2)
        
        # Reshape for batched matmul
        # q: [batch, num_heads, head_dim] -> [batch, num_heads, 1, head_dim]
        # k: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        q_reshaped = q.unsqueeze(2)  # [batch, heads, 1, dim]
        k_reshaped = sparse_k.transpose(1, 2)  # [batch, heads, seq, dim]
        v_reshaped = sparse_v.transpose(1, 2)  # [batch, heads, seq, dim]
        
        # Attention scores: [batch, heads, 1, seq]
        attn_weights = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        
        # Output: [batch, heads, 1, dim] -> [batch, heads, dim]
        o = torch.matmul(attn_weights, v_reshaped).squeeze(2)
        
        return o
    
    def _gather_kv_sparse(
        self,
        k_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: torch.Tensor,
        indices: torch.Tensor,  # [batch_size, k] token indices
        block_tables: torch.Tensor,  # [batch_size, max_blocks]
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather K/V vectors for given token indices from paged KV cache.
        
        Args:
            k_cache, v_cache: KV cache tensors
            indices: Token indices [batch_size, k]
            block_tables: Block tables [batch_size, max_blocks]
            block_size: Tokens per block
            
        Returns:
            k, v: Gathered tensors [batch_size, k, num_kv_heads, head_dim]
        """
        batch_size, k = indices.shape
        device = indices.device
        
        # Convert token indices to (block_idx, offset_in_block)
        block_indices = indices // block_size  # [batch, k]
        block_offsets = indices % block_size   # [batch, k]
        
        # Get physical block IDs from block_tables
        # block_tables: [batch, max_blocks]
        # block_indices: [batch, k] - indices into block_tables
        
        # Gather physical block IDs
        # Use advanced indexing: for each (batch, k), get block_tables[batch, block_indices[batch, k]]
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, k)
        physical_blocks = block_tables[batch_idx, block_indices.long()]  # [batch, k]
        
        # Compute flat cache indices: physical_block * block_size + offset
        flat_indices = physical_blocks.long() * block_size + block_offsets  # [batch, k]
        
        # Flatten cache for gathering
        # k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        # Reshape to [num_blocks * block_size, num_kv_heads, head_dim]
        cache_flat_k = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
        cache_flat_v = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])
        
        # Gather: [batch, k] indices -> [batch, k, num_kv_heads, head_dim]
        flat_indices_expanded = flat_indices.unsqueeze(-1).unsqueeze(-1)
        flat_indices_expanded = flat_indices_expanded.expand(-1, -1, cache_flat_k.shape[-2], cache_flat_k.shape[-1])
        
        # Use gather (need to handle batch dimension)
        gathered_k = torch.zeros(batch_size, k, cache_flat_k.shape[-2], cache_flat_k.shape[-1], 
                                device=device, dtype=k_cache.dtype)
        gathered_v = torch.zeros_like(gathered_k)
        
        for b in range(batch_size):
            gathered_k[b] = cache_flat_k[flat_indices[b].long()]
            gathered_v[b] = cache_flat_v[flat_indices[b].long()]
        
        return gathered_k, gathered_v
    
    def _gather_kv_range(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        start: int,
        end: int,
        block_tables: torch.Tensor,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather K/V for a contiguous range of tokens (for decode region).
        
        Args:
            k_cache, v_cache: KV cache tensors
            start, end: Token index range [start, end)
            block_tables: Block tables [batch_size, max_blocks]
            block_size: Tokens per block
            
        Returns:
            k, v: Gathered tensors [batch_size, end-start, num_kv_heads, head_dim]
        """
        batch_size = block_tables.shape[0]
        seq_len = end - start
        device = block_tables.device
        
        # Create indices for the range
        indices = torch.arange(start, end, device=device).unsqueeze(0).expand(batch_size, -1)
        
        return self._gather_kv_sparse(k_cache, v_cache, indices, block_tables, block_size)
