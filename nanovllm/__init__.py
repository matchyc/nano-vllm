# Lazy imports to avoid loading heavy dependencies (flash_attn, etc.) when not needed
def __getattr__(name):
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    elif name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams
        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["LLM", "SamplingParams"]
