"""
Benchmark script for sparse attention.

This script compares performance of dense vs sparse attention.
"""

import time
import torch
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def benchmark_sparse_attention(
    model_path: str,
    prompt: str,
    max_tokens: int = 50,
    use_sparse: bool = False,
    sparse_topk: int = 64,
    sparse_min_seq_len: int = 512,
):
    """
    Benchmark sparse vs dense attention.
    
    Args:
        model_path: Path to the model
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        use_sparse: Whether to use sparse attention
        sparse_topk: Top-k for sparse attention
        sparse_min_seq_len: Minimum sequence length to use sparse attention
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {'Sparse' if use_sparse else 'Dense'} Attention")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Prompt length: {len(prompt.split())} words")
    print(f"Max tokens: {max_tokens}")
    if use_sparse:
        print(f"Sparse topk: {sparse_topk}")
        print(f"Sparse min seq len: {sparse_min_seq_len}")
    print(f"{'='*60}\n")
    
    # Create LLM engine
    llm = LLM(
        model=model_path,
        use_sparse_attention=use_sparse,
        sparse_topk=sparse_topk,
        sparse_min_seq_len=sparse_min_seq_len,
    )
    
    # Prepare sampling params
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )
    
    # Warmup
    print("Warming up...")
    llm.generate([prompt[:100]], sampling_params, use_tqdm=False)
    
    # Benchmark
    print("\nRunning benchmark...")
    num_runs = 3
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage time: {avg_time:.2f}s")
    
    # Cleanup
    llm.exit()
    
    return avg_time


def main():
    """Run benchmark comparison."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python benchmark_sparse.py <model_path> [prompt]")
        print("\nExample:")
        print("  python benchmark_sparse.py /path/to/model 'Your prompt here'")
        sys.exit(1)
    
    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The quick brown fox jumps over the lazy dog. " * 50
    
    # Ensure prompt is long enough for sparse attention
    if len(prompt.split()) < 512:
        prompt = prompt * (512 // len(prompt.split()) + 1)
    
    print("=" * 60)
    print("Sparse Attention Benchmark")
    print("=" * 60)
    
    # Benchmark dense attention
    dense_time = benchmark_sparse_attention(
        model_path,
        prompt,
        max_tokens=50,
        use_sparse=False,
    )
    
    # Benchmark sparse attention
    sparse_time = benchmark_sparse_attention(
        model_path,
        prompt,
        max_tokens=50,
        use_sparse=True,
        sparse_topk=64,
        sparse_min_seq_len=512,
    )
    
    # Compare
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Dense attention:  {dense_time:.2f}s")
    print(f"Sparse attention: {sparse_time:.2f}s")
    speedup = dense_time / sparse_time if sparse_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
