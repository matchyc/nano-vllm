<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Prototype Sparse Attention (Experimental)

Nano-vLLM now includes an exploratory sparse-attention path that relies on an internal ANN index built directly from prefill keys. It is **single GPU only** and currently limits sparse lookups to prefill tokens (decode tokens are always attended densely over a short window).

Enable it by passing additional config fields when instantiating `LLM`:

```python
llm = LLM(
    "/YOUR/MODEL/PATH",
    use_sparse_attention=True,
    sparse_topk=64,
    sparse_min_seq_len=512,
    sparse_distance_metric="ip",
    sparse_decode_dense_window=128,
    sparse_ann_mode="exact",  # or "ivf"
)
```

Troubleshooting tips:

- Set `NANOVLLM_DISABLE_SPARSE_ATTENTION=1` to force the dense path at runtime.
- Choose `sparse_ann_mode="exact"` for brute-force correctness or `"ivf"` to enable a lightweight clustered ANN backend (see `sparse_ivf_*` flags for tuning).
- The current prototype only supports `sparse_index_granularity="layer_shared"`. Per-head indices are left as a future extension.

To compare latency with and without the sparse path on your hardware, use `python sparse_bench.py --model /path/to/Qwen3-0.6B --batch-size 4 --decode-tokens 64`.

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)