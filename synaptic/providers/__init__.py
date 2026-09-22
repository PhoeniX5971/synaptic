"""Provider SDKs are loaded only when their adapter is requested."""

from importlib import import_module

_ADAPTERS = {
    "ClaudeAdapter": ("claude", "claude"),
    "GeminiAdapter": ("gemini", "gemini"),
    "OpenAIAdapter": ("openai_", "openai"),
    "DeepSeekAdapter": ("deepseek", "deepseek"),
    "VertexAdapter": ("vertex", "vertex"),
    "TogetherAdapter": ("together", "together"),
    "UniversalLLMAdapter": ("universal_openai", "openai"),
    "XAIAdapter": ("xai", "xai"),
}
__all__ = list(_ADAPTERS)


def __getattr__(name):
    if name not in _ADAPTERS:
        raise AttributeError(name)
    module, extra = _ADAPTERS[name]
    try:
        return getattr(import_module(f".{module}", __name__), name)
    except ImportError as exc:
        raise ImportError(f"Install provider support: pip install 'synaptic[{extra}]'") from exc
