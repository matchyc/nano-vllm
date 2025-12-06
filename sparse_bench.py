import argparse
import time

from nanovllm import LLM, SamplingParams


def run_once(args, use_sparse: bool):
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_context,
        use_sparse_attention=use_sparse,
        sparse_topk=args.sparse_topk,
        sparse_min_seq_len=args.min_seq_len,
        sparse_distance_metric=args.metric,
        sparse_decode_dense_window=args.decode_window,
    )
    prompts = [args.prompt] * args.batch_size
    sampling = SamplingParams(max_tokens=args.decode_tokens, temperature=0.0, ignore_eos=True)
    llm.generate(["Warmup"], SamplingParams(max_tokens=1))
    t0 = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    elapsed = time.time() - t0
    llm.exit()
    total_tokens = args.batch_size * args.decode_tokens
    return elapsed, total_tokens / max(elapsed, 1e-6), outputs[0]["text"]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare dense vs sparse attention latency.")
    parser.add_argument("--model", required=True, help="Path to HF checkpoint compatible with nano-vllm.")
    parser.add_argument("--prompt", default="Prototype sparse attention benchmark.", help="Prompt to replicate.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--sparse-topk", type=int, default=128)
    parser.add_argument("--min-seq-len", type=int, default=512)
    parser.add_argument("--decode-window", type=int, default=128)
    parser.add_argument("--metric", choices=["ip", "l2"], default="ip")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dense_time, dense_tp, _ = run_once(args, use_sparse=False)
    sparse_time, sparse_tp, sample = run_once(args, use_sparse=True)
    print(f"Dense   : {dense_time:.2f}s, {dense_tp:.2f} tok/s")
    print(f"Sparse  : {sparse_time:.2f}s, {sparse_tp:.2f} tok/s, sample='{sample[:120]}'")


if __name__ == "__main__":
    main()
