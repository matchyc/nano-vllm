#!/usr/bin/env python3
"""
Example script demonstrating sparse attention in nano-vllm.

This script shows how to:
1. Enable sparse attention via config
2. Run inference with dense attention (baseline)
3. Run inference with sparse attention
4. Compare outputs and timing

Requirements:
- CUDA GPU
- flash_attn installed
- Model weights (e.g., Qwen3-0.6B)

Usage:
    python example_sparse.py --model /path/to/Qwen3-0.6B
"""

import os
import sys
import argparse
import time

# Import LLM and SamplingParams using explicit imports to ensure lazy loading works
def main():
    parser = argparse.ArgumentParser(description="Sparse attention demo")
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
                       help="Path to model weights")
    parser.add_argument("--sparse-topk", type=int, default=64,
                       help="Number of top keys for sparse attention")
    parser.add_argument("--sparse-min-seq", type=int, default=256,
                       help="Minimum sequence length for sparse attention")
    parser.add_argument("--prompt-tokens", type=int, default=512,
                       help="Approximate number of prompt tokens")
    parser.add_argument("--max-tokens", type=int, default=64,
                       help="Maximum tokens to generate")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Nano-vLLM Sparse Attention Demo")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.isdir(args.model):
        print(f"\nError: Model path does not exist: {args.model}")
        print("\nTo run this demo, you need:")
        print("  1. Download a Qwen3 model (e.g., Qwen3-0.6B)")
        print("  2. Specify the path: --model /path/to/model")
        print("\nExample:")
        print("  python example_sparse.py --model ~/huggingface/Qwen3-0.6B/")
        return
    
    try:
        from nanovllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"\nError importing required modules: {e}")
        print("\nMake sure you have installed:")
        print("  - torch")
        print("  - transformers")
        print("  - flash_attn (requires CUDA)")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Create a long prompt
    base_prompt = "Please provide a detailed explanation of how transformers work in deep learning. Cover the following topics: attention mechanism, self-attention, multi-head attention, positional encoding, encoder-decoder architecture, and their applications in NLP. "
    # Repeat to get desired length
    long_prompt = base_prompt
    while len(tokenizer.encode(long_prompt)) < args.prompt_tokens:
        long_prompt += base_prompt
    
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": long_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"\nPrompt length: {prompt_tokens} tokens")
    print(f"Generation length: {args.max_tokens} tokens")
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    
    # Run with dense attention (baseline)
    print("\n" + "-" * 60)
    print("1. Dense Attention (Baseline)")
    print("-" * 60)
    
    llm_dense = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        use_sparse_attention=False,  # Disabled
    )
    
    start = time.perf_counter()
    outputs_dense = llm_dense.generate([prompt], sampling_params, use_tqdm=False)
    dense_time = time.perf_counter() - start
    
    print(f"Time: {dense_time:.2f}s")
    print(f"Generated: {outputs_dense[0]['text'][:200]}...")
    
    # Cleanup
    llm_dense.exit()
    del llm_dense
    
    # Run with sparse attention
    print("\n" + "-" * 60)
    print("2. Sparse Attention")
    print("-" * 60)
    print(f"   TopK: {args.sparse_topk}")
    print(f"   Min seq len: {args.sparse_min_seq}")
    
    llm_sparse = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        use_sparse_attention=True,
        sparse_topk=args.sparse_topk,
        sparse_min_seq_len=args.sparse_min_seq,
        sparse_distance_metric="ip",
        sparse_index_granularity="layer_shared",
        sparse_include_decode_dense=True,
        sparse_max_decode_tokens=32,
        sparse_debug=True,
        # MLANN algorithm parameters
        sparse_mlann_k_train=32,    # k-NN for training labels
        sparse_mlann_n_trees=8,     # Number of RP trees
        sparse_mlann_max_depth=8,   # Max tree depth
        sparse_mlann_min_leaf_size=10,
    )
    
    start = time.perf_counter()
    outputs_sparse = llm_sparse.generate([prompt], sampling_params, use_tqdm=False)
    sparse_time = time.perf_counter() - start
    
    print(f"Time: {sparse_time:.2f}s")
    print(f"Generated: {outputs_sparse[0]['text'][:200]}...")
    
    # Cleanup
    llm_sparse.exit()
    del llm_sparse
    
    # Compare
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Dense time:  {dense_time:.2f}s")
    print(f"Sparse time: {sparse_time:.2f}s")
    print(f"Speedup: {dense_time/sparse_time:.2f}x")
    
    # Check if outputs match (they may differ due to sparse approximation)
    dense_tokens = outputs_dense[0]['token_ids']
    sparse_tokens = outputs_sparse[0]['token_ids']
    matching = sum(1 for a, b in zip(dense_tokens, sparse_tokens) if a == b)
    print(f"\nToken match: {matching}/{len(dense_tokens)} ({100*matching/len(dense_tokens):.1f}%)")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
