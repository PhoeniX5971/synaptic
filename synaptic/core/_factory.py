from typing import Any, List, Optional

from ..providers import (
    ClaudeAdapter,
    DeepSeekAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    TogetherAdapter,
    UniversalLLMAdapter,
    VertexAdapter,
    XAIAdapter,
)
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
        return ClaudeAdapter(**kwargs, max_tokens=max_tokens)
    elif provider == Provider.OPENAI:
        return OpenAIAdapter(**kwargs)
    elif provider == Provider.GEMINI:
        return GeminiAdapter(**kwargs)
    elif provider == Provider.VERTEX:
        return VertexAdapter(
            **{**kwargs, "api_key": None},
            location=location,
            project=project,
        )
    elif provider == Provider.DEEPSEEK:
        return DeepSeekAdapter(**kwargs)
    elif provider == Provider.TOGETHER:
        return TogetherAdapter(**kwargs)
    elif provider == Provider.XAI:
        return XAIAdapter(**kwargs)
    elif provider == Provider.UNIVERSAL_OPENAI:
        return UniversalLLMAdapter(**kwargs, base_url=base_url or "")
    else:
        return GeminiAdapter(**kwargs)
