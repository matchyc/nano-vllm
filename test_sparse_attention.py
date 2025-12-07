"""
Test script for sparse attention implementation.

This script tests the basic functionality of sparse attention components.
"""

import torch
import numpy as np
from nanovllm.sparse.mlann_index import MLANNIndex
from nanovllm.sparse.sparse_attention import sparse_attention_decode


def test_mlann_index():
    """Test MLANN index build and query."""
    print("Testing MLANN index...")
    
    # Create random corpus
    num_tokens = 1000
    dim = 128
    corpus = np.random.randn(num_tokens, dim).astype(np.float32)
    
    # Build index
    index = MLANNIndex(metric="ip", index_type="PCA")
    index.build(corpus, n_trees=10, depth=6, votes_required=5)
    print(f"✓ Index built successfully (corpus size: {num_tokens}, dim: {dim})")
    
    # Query
    num_queries = 10
    queries = np.random.randn(num_queries, dim).astype(np.float32)
    k = 64
    indices = index.query(queries, k)
    
    assert indices.shape == (num_queries, k), f"Expected shape ({num_queries}, {k}), got {indices.shape}"
    assert indices.dtype == np.int64, f"Expected int64, got {indices.dtype}"
    assert np.all(indices >= 0) and np.all(indices < num_tokens), "Indices out of range"
    print(f"✓ Query successful (queries: {num_queries}, k: {k})")
    
    # Test torch interface
    queries_torch = torch.randn(num_queries, dim)
    indices_torch = index.query_torch(queries_torch, k)
    
    assert indices_torch.shape == (num_queries, k), f"Expected shape ({num_queries}, {k}), got {indices_torch.shape}"
    assert indices_torch.dtype == torch.int64, f"Expected int64, got {indices_torch.dtype}"
    print(f"✓ Torch interface works correctly")
    
    print("MLANN index test passed!\n")


def test_sparse_attention_decode():
    """Test sparse attention decode computation."""
    print("Testing sparse attention decode...")
    
    batch_size = 2
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    topk = 32
    block_size = 256
    num_blocks = 4
    
    # Create random tensors
    q = torch.randn(batch_size, num_heads, head_dim)
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
    
    # Create random indices (token positions within sequence)
    indices = torch.randint(0, num_blocks * block_size, (batch_size, topk))
    
    # Block table: logical -> physical (identity mapping for test)
    block_table = list(range(num_blocks))
    
    scale = 1.0 / np.sqrt(head_dim)
    
    # Compute sparse attention
    output = sparse_attention_decode(
        q,
        k_cache,
        v_cache,
        indices,
        block_table,
        block_size,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
    )
    
    assert output.shape == (batch_size, num_heads, head_dim), \
        f"Expected shape ({batch_size}, {num_heads}, {head_dim}), got {output.shape}"
    print(f"✓ Sparse attention decode successful (output shape: {output.shape})")
    
    print("Sparse attention decode test passed!\n")


def test_correctness_small_case():
    """Test that sparse attention matches dense attention for small k >= seq_len."""
    print("Testing correctness (sparse vs dense for k >= seq_len)...")
    
    batch_size = 1
    num_heads = 4
    num_kv_heads = 2
    head_dim = 32
    seq_len = 16
    topk = seq_len  # Use all tokens
    
    # Create deterministic tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, head_dim)
    k = torch.randn(seq_len, num_kv_heads, head_dim)
    v = torch.randn(seq_len, num_kv_heads, head_dim)
    
    # Create cache (single block)
    block_size = 256
    k_cache = torch.zeros(1, block_size, num_kv_heads, head_dim)
    v_cache = torch.zeros(1, block_size, num_kv_heads, head_dim)
    k_cache[0, :seq_len] = k
    v_cache[0, :seq_len] = v
    
    # Indices: use all tokens
    indices = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]
    block_table = [0]
    
    scale = 1.0 / np.sqrt(head_dim)
    
    # Sparse attention
    output_sparse = sparse_attention_decode(
        q,
        k_cache,
        v_cache,
        indices,
        block_table,
        block_size,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
    )
    
    # Dense attention (manual computation)
    # Expand q: [1, num_heads, 1, head_dim]
    q_expanded = q.unsqueeze(2)
    # Expand k/v: [1, 1, seq_len, num_kv_heads, head_dim]
    k_expanded = k.unsqueeze(0).unsqueeze(0)
    v_expanded = v.unsqueeze(0).unsqueeze(0)
    
    # Repeat kv heads to match q heads
    repeat_factor = num_heads // num_kv_heads
    k_expanded = k_expanded.repeat(1, repeat_factor, 1, 1, 1)
    v_expanded = v_expanded.repeat(1, repeat_factor, 1, 1, 1)
    
    # Reshape: [1, num_heads, seq_len, head_dim]
    k_expanded = k_expanded.squeeze(1)
    v_expanded = v_expanded.squeeze(1)
    
    # Compute attention
    scores = torch.matmul(q_expanded, k_expanded.transpose(-2, -1)) * scale
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    output_dense = torch.matmul(attn_weights, v_expanded).squeeze(2)
    
    # Compare
    max_diff = torch.max(torch.abs(output_sparse - output_dense)).item()
    print(f"  Max difference: {max_diff:.6f}")
    
    # Allow small numerical differences
    assert max_diff < 1e-5, f"Difference too large: {max_diff}"
    print(f"✓ Sparse and dense attention match (max diff: {max_diff:.6f})")
    
    print("Correctness test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Sparse Attention Implementation")
    print("=" * 60 + "\n")
    
    try:
        test_mlann_index()
        test_sparse_attention_decode()
        test_correctness_small_case()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
