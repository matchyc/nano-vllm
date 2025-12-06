__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    if name == "LLM":
        from .llm import LLM as _LLM
        return _LLM
    if name == "SamplingParams":
        from .sampling_params import SamplingParams as _SamplingParams
        return _SamplingParams
    raise AttributeError(name)
