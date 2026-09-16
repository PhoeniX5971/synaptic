from ._request import XAIRequest
import inspect
import asyncio
import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, List, Optional


from xai_sdk.aio.client import Client as AsyncClient
from xai_sdk.sync.client import Client as SyncClient
from xai_sdk.chat import (
    tool as xai_tool,
)
from xai_sdk.proto import chat_pb2

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import Tool, ToolRegistry, collect_tools, register_callback
from .helpers import parse_calls


_STRUCTURED_OUTPUT_TOOL = "__structured_output__"


def _make_xai_tool(t: Tool) -> chat_pb2.Tool:
    decl = t.declaration
    return xai_tool(
        name=decl["name"],
        description=decl.get("description", ""),
        parameters=decl.get("parameters", {"type": "object", "properties": {}}),
    )


from ...core.capabilities import Capabilities


class XAIAdapter(XAIRequest, BaseModel):
    capabilities = Capabilities(images=True, audio=False)

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
        self.model = model
        self.api_key = api_key or None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.instructions = instructions
        self.tool_registry = tool_registry

        self.bound_tools = list(tools or [])
        self.synaptic_tools = list(tools or [])
        self.xai_tools: List[chat_pb2.Tool] = []

        self._sync_client = SyncClient(api_key=self.api_key)
        self._async_client = AsyncClient(api_key=self.api_key)

        self._subscription = register_callback(self._invalidate_tools, tool_registry)
        self._invalidate_tools()

    def _invalidate_tools(self) -> None:
        self._convert_tools()

    def _convert_tools(self) -> None:
        all_tools = collect_tools(self.bound_tools, self.tool_registry)
        self.xai_tools = [_make_xai_tool(t) for t in all_tools.values()]
        self.synaptic_tools = list(all_tools.values())

    def _parse_response(self, response: Any, usage: Any = None) -> ResponseMem:
        created = datetime.now().astimezone(timezone.utc)
        message = response.content or ""
        tool_calls = parse_calls(response)
        if self.response_format == ResponseFormat.JSON and not tool_calls:
            try:
                message = json.dumps(json.loads(message))
            except Exception:
                pass
        u = usage or getattr(response, "usage", None)
        return ResponseMem(
            message=message,
            created=created,
            tool_calls=tool_calls,
            input_tokens=getattr(u, "prompt_tokens", None),
            output_tokens=getattr(u, "completion_tokens", None),
            finish_reason=str(getattr(response, "finish_reason", "")) or None,
            extensions={"response_id": getattr(response, "id", None)},
        )

    def invoke(
        self,
        prompt: Optional[str],
        role: str = "user",
        images: Optional[List[Any]] = None,
        audio: Optional[List[Any]] = None,
        **kwargs,
    ) -> ResponseMem:
        self.capabilities.check("invoke", images=images, audio=audio)
        chat = self._create_chat(self._sync_client, prompt, images, audio, **kwargs)
        response = chat.sample()
        return self._parse_response(response)

    async def astream(
        self,
        prompt: Optional[str],
        role: str = "user",
        images: Optional[List[Any]] = None,
        audio: Optional[List[Any]] = None,
        abort=None,
        **kwargs,
    ) -> AsyncIterator[ResponseChunk]:
        self.capabilities.check("astream", images=images, audio=audio)
        chat = self._create_chat(self._async_client, prompt, images, audio, **kwargs)

        accumulated = ""
        last_response = None
        source = chat.stream()
        try:
            async for response, chunk in source:
                if abort and abort.is_set():
                    raise asyncio.CancelledError()
                last_response = response
                for choice in chunk.choices:
                    piece = choice.content
                    if piece:
                        accumulated += piece
                        yield ResponseChunk(text=piece, is_final=False, function_call=None)
        finally:
            close = getattr(source, "aclose", None) or getattr(source, "close", None)
            if close:
                result = close()
                if inspect.isawaitable(result):
                    await result

        for tfc in parse_calls(last_response) if last_response is not None else []:
            yield ResponseChunk(text="", is_final=False, function_call=tfc)

        u = getattr(last_response, "usage", None)
        yield ResponseChunk(
            text=accumulated,
            is_final=True,
            function_call=None,
            input_tokens=getattr(u, "prompt_tokens", None),
            output_tokens=getattr(u, "completion_tokens", None),
            finish_reason=str(getattr(last_response, "finish_reason", "")) or None,
            extensions={"response_id": getattr(last_response, "id", None)},
        )
