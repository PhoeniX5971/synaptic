from typing import Any, List, Optional

from .. import providers
from .base import History, ResponseFormat
from .provider import Provider
from .tool import Tool, ToolRegistry


def build_adapter(
    provider: Provider,
    model: str,
    temperature: float,
    api_key: str,
    max_tokens: int,
    tools: Optional[List[Tool]],
    history: History,
    response_format: ResponseFormat,
    response_schema: Any,
    instructions: str,
    location: Optional[str],
    project: Optional[str],
    base_url: Optional[str],
    tool_registry: Optional[ToolRegistry] = None,
) -> Any:
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        history=history,
        api_key=api_key,
        response_format=response_format,
        response_schema=response_schema,
        instructions=instructions,
        tool_registry=tool_registry,
    )

    if provider == Provider.CLAUDE:
        return _adapter("ClaudeAdapter")(**kwargs)
    elif provider == Provider.OPENAI:
        return _adapter("OpenAIAdapter")(**kwargs, base_url=base_url)
    elif provider == Provider.GEMINI:
        return _adapter("GeminiAdapter")(**kwargs)
    elif provider == Provider.VERTEX:
        return _adapter("VertexAdapter")(
            **{**kwargs, "api_key": None},
            location=location,
            project=project,
        )
    elif provider == Provider.DEEPSEEK:
        return _adapter("DeepSeekAdapter")(**kwargs, base_url=base_url)
    elif provider == Provider.TOGETHER:
        return _adapter("TogetherAdapter")(**kwargs)
    elif provider == Provider.XAI:
        return _adapter("XAIAdapter")(**kwargs)
    elif provider == Provider.UNIVERSAL_OPENAI:
        return _adapter("UniversalLLMAdapter")(**kwargs, base_url=base_url or "")
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


def __getattr__(name):
    return getattr(providers, name)


def _adapter(name):
    return globals()[name] if name in globals() else getattr(providers, name)
