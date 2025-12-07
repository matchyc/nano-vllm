#!/usr/bin/env python3
"""
Benchmark script for sparse attention in nano-vllm.

This script benchmarks:
1. ANN index build time vs sequence length
2. ANN query time vs topk
3. IVF vs exact mode comparison
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

from nanovllm.sparse.ann_index import ANNIndex
from nanovllm.sparse.manager import SparseAttentionManager


def format_time(ms: float) -> str:
    """Format time in appropriate units."""
    if ms < 1:
        return f"{ms * 1000:.2f} μs"
    elif ms < 1000:
        return f"{ms:.2f} ms"
    else:
        return f"{ms / 1000:.2f} s"


def bench_ann_index_build():
    """Benchmark ANN index build time vs sequence length."""
    print("\n" + "=" * 60)
    print("Benchmark: ANN Index Build Time")
    print("=" * 60)
    
    dim = 128  # Typical: num_kv_heads * head_dim = 4 * 32 = 128
    seq_lengths = [256, 512, 1024, 2048, 4096]
    modes = ["exact", "ivf"]
    
    results: Dict[str, List[float]] = {mode: [] for mode in modes}
    
    print(f"\nDimension: {dim}")
    print(f"{'Seq Length':<12} {'Exact':<15} {'IVF (nlist=64)':<15}")
    print("-" * 45)
    
    for seq_len in seq_lengths:
        np.random.seed(42)
        corpus = np.random.randn(seq_len, dim).astype(np.float32)
        
        row = f"{seq_len:<12}"
        
        for mode in modes:
            index = ANNIndex(
                metric="ip", 
                mode=mode,
                nlist=64 if mode == "ivf" else 64,
                nprobe=8,
            )
            
            # Warmup
            index.build(corpus)
            
            # Benchmark
            n_runs = 5
            times = []
            for _ in range(n_runs):
                index = ANNIndex(metric="ip", mode=mode, nlist=64, nprobe=8)
                start = time.perf_counter()
                index.build(corpus)
                times.append((time.perf_counter() - start) * 1000)
            
            avg_time = np.mean(times)
            results[mode].append(avg_time)
            row += f"{format_time(avg_time):<15}"
        
        print(row)
    
    return results


def bench_ann_query():
    """Benchmark ANN query time vs topk."""
    print("\n" + "=" * 60)
    print("Benchmark: ANN Query Time")
    print("=" * 60)
    
    dim = 128
    seq_len = 2048
    n_queries = 8  # Typical batch size
    topks = [16, 32, 64, 128, 256]
    modes = ["exact", "ivf"]
    
    np.random.seed(42)
    corpus = np.random.randn(seq_len, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    print(f"\nCorpus size: {seq_len}, Query batch: {n_queries}, Dimension: {dim}")
    print(f"{'TopK':<12} {'Exact':<15} {'IVF':<15}")
    print("-" * 45)
    
    # Build indices once
    exact_index = ANNIndex(metric="ip", mode="exact")
    exact_index.build(corpus)
    
    ivf_index = ANNIndex(metric="ip", mode="ivf", nlist=64, nprobe=8)
    ivf_index.build(corpus)
    
    for k in topks:
        row = f"{k:<12}"
        
        # Exact
        n_runs = 10
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            exact_index.query(queries, k)
            times.append((time.perf_counter() - start) * 1000)
        row += f"{format_time(np.mean(times)):<15}"
        
        # IVF
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            ivf_index.query(queries, k)
            times.append((time.perf_counter() - start) * 1000)
        row += f"{format_time(np.mean(times)):<15}"
        
        print(row)


def bench_ivf_recall():
    """Benchmark IVF recall vs nprobe."""
    print("\n" + "=" * 60)
    print("Benchmark: IVF Recall vs nprobe")
    print("=" * 60)
    
    dim = 128
    seq_len = 4096
    n_queries = 100
    k = 64
    nlist = 64
    nprobes = [1, 2, 4, 8, 16, 32]
    
    np.random.seed(42)
    corpus = np.random.randn(seq_len, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    # Ground truth
    exact_index = ANNIndex(metric="ip", mode="exact")
    exact_index.build(corpus)
    gt_indices = exact_index.query(queries, k)
    
    print(f"\nCorpus: {seq_len}, Queries: {n_queries}, k: {k}, nlist: {nlist}")
    print(f"{'nprobe':<12} {'Recall@k':<15} {'Query Time':<15}")
    print("-" * 45)
    
    for nprobe in nprobes:
        ivf_index = ANNIndex(metric="ip", mode="ivf", nlist=nlist, nprobe=nprobe)
        ivf_index.build(corpus)
        
        # Query
        start = time.perf_counter()
        ivf_indices = ivf_index.query(queries, k)
        query_time = (time.perf_counter() - start) * 1000
        
        # Compute recall
        recalls = []
        for i in range(n_queries):
            gt_set = set(gt_indices[i].tolist())
            pred_set = set(ivf_indices[i].tolist())
            recalls.append(len(gt_set & pred_set) / k)
        
        avg_recall = np.mean(recalls)
        print(f"{nprobe:<12} {avg_recall:.2%}{'':>8} {format_time(query_time):<15}")


def bench_sparse_manager():
    """Benchmark SparseAttentionManager end-to-end."""
    print("\n" + "=" * 60)
    print("Benchmark: SparseAttentionManager (Full Pipeline)")
    print("=" * 60)
    
    class MockConfig:
        use_sparse_attention = True
        sparse_topk = 64
        sparse_min_seq_len = 256
        sparse_distance_metric = "ip"
        sparse_index_granularity = "layer_shared"
        sparse_ann_mode = "exact"
        sparse_ivf_nlist = 64
        sparse_ivf_nprobe = 8
        sparse_include_decode_dense = True
        sparse_max_decode_tokens = 32
        sparse_debug = False
        kvcache_block_size = 256
    
    num_layers = 28
    num_kv_heads = 4
    head_dim = 128
    block_size = 256
    
    seq_lengths = [512, 1024, 2048, 4096]
    
    print(f"\nLayers: {num_layers}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}")
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
    parser = argparse.ArgumentParser(description="Benchmark sparse attention")
    parser.add_argument("--full", action="store_true", 
                       help="Run full end-to-end benchmark (requires GPU and model)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sparse Attention Benchmarks")
    print("=" * 60)
    
    # CPU-based benchmarks
    bench_ann_index_build()
    bench_ann_query()
    bench_ivf_recall()
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
