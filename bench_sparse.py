#!/usr/bin/env python3
"""
Benchmark script for sparse attention in nano-vllm (MLANN-based).

This script benchmarks:
1. MLANN index build time vs sequence length
2. MLANN query time vs topk
3. Effect of n_trees on recall
4. (Optional, requires GPU + model) Full end-to-end latency comparison

Usage:
    python bench_sparse.py [--full]  # --full requires GPU and model
"""

import sys
sys.path.insert(0, '/workspace')

import argparse
import time
import numpy as np
import torch
from typing import Dict, List

from nanovllm.sparse.mlann_index import MLANNIndex, compute_recall
from nanovllm.sparse.manager import SparseAttentionManager


def format_time(ms: float) -> str:
    """Format time in appropriate units."""
    if ms < 1:
        return f"{ms * 1000:.2f} μs"
    elif ms < 1000:
        return f"{ms:.2f} ms"
    else:
        return f"{ms / 1000:.2f} s"


def bench_mlann_index_build():
    """Benchmark MLANN index build time vs sequence length."""
    print("\n" + "=" * 60)
    print("Benchmark: MLANN Index Build Time")
    print("=" * 60)
    
    dim = 128  # Typical: num_kv_heads * head_dim = 4 * 32 = 128
    seq_lengths = [256, 512, 1024, 2048, 4096]
    n_trees_options = [4, 8, 16]
    
    print(f"\nDimension: {dim}, k_train: 32, max_depth: 8")
    header = f"{'Seq Length':<12}"
    for n_trees in n_trees_options:
        header += f"{n_trees} trees{'':>6}"
    print(header)
    print("-" * (12 + 15 * len(n_trees_options)))
    
    for seq_len in seq_lengths:
        np.random.seed(42)
        corpus = np.random.randn(seq_len, dim).astype(np.float32)
        corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
        
        row = f"{seq_len:<12}"
        
        for n_trees in n_trees_options:
            # Benchmark
            n_runs = 3
            times = []
            for _ in range(n_runs):
                index = MLANNIndex(metric="ip", k_train=32, n_trees=n_trees, max_depth=8)
                start = time.perf_counter()
                index.build(corpus)
                times.append((time.perf_counter() - start) * 1000)
            
            avg_time = np.mean(times)
            row += f"{format_time(avg_time):<15}"
        
        print(row)


def bench_mlann_query():
    """Benchmark MLANN query time vs topk."""
    print("\n" + "=" * 60)
    print("Benchmark: MLANN Query Time")
    print("=" * 60)
    
    dim = 128
    seq_len = 2048
    n_queries = 8  # Typical batch size
    topks = [16, 32, 64, 128, 256]
    n_trees_options = [4, 8, 16]
    
    np.random.seed(42)
    corpus = np.random.randn(seq_len, dim).astype(np.float32)
    corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    queries = corpus[:n_queries]  # Same distribution queries
    
    print(f"\nCorpus size: {seq_len}, Query batch: {n_queries}, Dimension: {dim}")
    
    # Build indices once
    indices_dict = {}
    for n_trees in n_trees_options:
        index = MLANNIndex(metric="ip", k_train=32, n_trees=n_trees, max_depth=8)
        index.build(corpus)
        indices_dict[n_trees] = index
    
    header = f"{'TopK':<12}"
    for n_trees in n_trees_options:
        header += f"{n_trees} trees{'':>6}"
    print(header)
    print("-" * (12 + 15 * len(n_trees_options)))
    
    for k in topks:
        row = f"{k:<12}"
        
        for n_trees in n_trees_options:
            n_runs = 10
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                indices_dict[n_trees].query(queries, k)
                times.append((time.perf_counter() - start) * 1000)
            row += f"{format_time(np.mean(times)):<15}"
        
        print(row)


def bench_mlann_recall():
    """Benchmark MLANN recall vs n_trees."""
    print("\n" + "=" * 60)
    print("Benchmark: MLANN Recall vs n_trees")
    print("=" * 60)
    
    dim = 128
    seq_len = 2048
    n_queries = 50
    k = 64
    n_trees_options = [1, 2, 4, 8, 16, 32]
    
    np.random.seed(42)
    corpus = np.random.randn(seq_len, dim).astype(np.float32)
    corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    queries = corpus[:n_queries]  # Same distribution
    
    # Ground truth
    gt_indices = np.argsort(-(queries @ corpus.T), axis=1)[:, :k]
    
    print(f"\nCorpus: {seq_len}, Queries: {n_queries}, k: {k}")
    print(f"{'n_trees':<12} {'Recall@k':<15} {'Build Time':<15} {'Query Time':<15}")
    print("-" * 60)
    
    for n_trees in n_trees_options:
        index = MLANNIndex(metric="ip", k_train=32, n_trees=n_trees, max_depth=8)
        
        # Build
        start = time.perf_counter()
        index.build(corpus)
        build_time = (time.perf_counter() - start) * 1000
        
        # Query
        start = time.perf_counter()
        mlann_indices = index.query(queries, k)
        query_time = (time.perf_counter() - start) * 1000
        
        # Compute recall
        recall = compute_recall(mlann_indices, gt_indices)
        print(f"{n_trees:<12} {recall:.2%}{'':>8} {format_time(build_time):<15} {format_time(query_time):<15}")


def bench_sparse_manager():
    """Benchmark SparseAttentionManager end-to-end with MLANN."""
    print("\n" + "=" * 60)
    print("Benchmark: SparseAttentionManager (MLANN Pipeline)")
    print("=" * 60)
    
    class MockConfig:
        use_sparse_attention = True
        sparse_topk = 64
        sparse_min_seq_len = 256
        sparse_distance_metric = "ip"
        sparse_index_granularity = "layer_shared"
        sparse_include_decode_dense = True
        sparse_max_decode_tokens = 32
        sparse_debug = False
        kvcache_block_size = 256
        # MLANN config
        sparse_mlann_k_train = 32
        sparse_mlann_n_trees = 8
        sparse_mlann_max_depth = 8
        sparse_mlann_min_leaf_size = 10
    
    num_layers = 28
    num_kv_heads = 4
    head_dim = 128
    block_size = 256
    
    seq_lengths = [512, 1024, 2048, 4096]
    
    print(f"\nLayers: {num_layers}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}")
    print(f"MLANN: k_train=32, n_trees=8, max_depth=8")
    print(f"{'Seq Length':<12} {'Index Build':<15} {'Query (×{num_layers})':<20}")
    print("-" * 50)
    
    for seq_len in seq_lengths:
        num_blocks = (seq_len + block_size - 1) // block_size
        
        # Create mock KV cache
        kv_cache = torch.randn(2, num_layers, num_blocks + 10, block_size, num_kv_heads, head_dim)
        block_table = list(range(num_blocks))
        
        manager = SparseAttentionManager(
            config=MockConfig(),
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        
        # Benchmark index build
        start = time.perf_counter()
        manager.build_indices_from_kv_cache(
            kv_cache, [seq_len], [block_table], block_size
        )
        build_time = (time.perf_counter() - start) * 1000
        
        # Benchmark queries (simulate decode)
        batch_size = 1
        num_heads = 8  # Query heads
        queries = torch.randn(batch_size, num_heads, head_dim)
        
        start = time.perf_counter()
        for layer_id in range(num_layers):
            manager.query(layer_id, queries)
        query_time = (time.perf_counter() - start) * 1000
        
        print(f"{seq_len:<12} {format_time(build_time):<15} {format_time(query_time):<20}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark sparse attention with MLANN")
    parser.add_argument("--full", action="store_true", 
                       help="Run full end-to-end benchmark (requires GPU and model)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sparse Attention Benchmarks (MLANN)")
    print("=" * 60)
    
    # CPU-based benchmarks
    bench_mlann_index_build()
    bench_mlann_query()
    bench_mlann_recall()
    bench_sparse_manager()
    
    if args.full:
        print("\n" + "=" * 60)
        print("Full End-to-End Benchmark")
        print("=" * 60)
        print("\nNote: Full benchmark requires:")
        print("  - CUDA GPU")
        print("  - flash_attn installed")
        print("  - Model weights (e.g., Qwen3-0.6B)")
        print("\nTo run full benchmark:")
        print("  1. Ensure model path in example_sparse.py")
        print("  2. Run: python example_sparse.py")
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
