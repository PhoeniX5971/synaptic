import inspect
from .._bridge import stream
from .._chat import assemble, response_memory
from ...run.scheduling import ThreadScheduler
from ...core.capabilities import Capabilities
from typing import Any, Dict, List, Optional

from together import Together

from ...core.base import BaseModel, History, ResponseFormat, ResponseMem
from ...core.tool import Tool, ToolRegistry, collect_tools, register_callback
from .helpers import messages


class TogetherAdapter(BaseModel):
    capabilities = Capabilities(images=True, audio=True, async_transport="bounded-thread")

    def __init__(
        self,
        model: str,
        history: History | None,
        api_key: str,
        response_format: ResponseFormat,
        response_schema: Any,
        tools: Optional[List[Tool]],
        temperature: float = 0.8,
        instructions: str = "",
        tool_registry: Optional[ToolRegistry] = None,
        max_tokens: int = 1024,
    ):
        self.client = Together(api_key=api_key)
        self._scheduler = ThreadScheduler(1)
        self.model = model
        self.history = history
        self.bound_tools = list(tools or [])
        self.synaptic_tools = list(tools or [])
        self.together_tools: List[Dict[str, Any]] = []
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.response_format = response_format
        self.response_schema = response_schema
        self.instructions = instructions
        self.tool_registry = tool_registry
        self.role_map = {"user": "user", "assistant": "assistant", "system": "system"}
        self._subscription = register_callback(self._invalidate_tools, tool_registry)
        self._invalidate_tools()

    def _invalidate_tools(self) -> None:
        self._convert_tools()

    def _convert_tools(self) -> None:
        self.together_tools = []
        all_tools = collect_tools(self.bound_tools, self.tool_registry)
        for tool in all_tools.values():
            self.together_tools.append({"type": "function", "function": tool.declaration})
        self.synaptic_tools = list(all_tools.values())

    def _request(self, prompt, role, images, audio, should_stream, kwargs) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages(self, prompt, role, images, audio),
            "temperature": self.temperature,
            **kwargs,
        }
        if should_stream:
            params["stream"] = True
        if self.together_tools and self.response_format == ResponseFormat.NONE:
            params["tools"] = self.together_tools
            params["tool_choice"] = "auto"
        if self.response_format == ResponseFormat.JSON:
            params["response_format"] = {"type": "json_object"}
        return params

    def invoke(
        self,
        prompt: Optional[str],
        role: str = "user",
        images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        **kwargs,
    ) -> ResponseMem:
        response = self.client.chat.completions.create(
            **self._request(prompt, role, images, audio, False, kwargs)
        )
        return response_memory(response)

    async def astream(self, prompt, role="user", images=None, audio=None, abort=None, **kwargs):
        request = self._request(prompt, role, images, audio, True, kwargs)
        source = stream(lambda: self.client.chat.completions.create(**request), self._scheduler)
        source = assemble(source, abort)
        try:
            async for chunk in source:
                yield chunk
        finally:
            close = getattr(source, "aclose", None) or getattr(source, "close", None)
            if close:
                result = close()
                if inspect.isawaitable(result):
                    await result
