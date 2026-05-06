import asyncio
import json
import threading
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from dotenv import load_dotenv
from together import Together

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import Tool, ToolCall, ToolRegistry, collect_tools, register_callback
from .helpers import messages, parse_calls, pending_calls

load_dotenv()


class TogetherAdapter(BaseModel):
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
    ):
        self.client = Together(api_key=api_key)
        self.model = model
        self.history = history
        self.synaptic_tools = list(tools or [])
        self.together_tools: List[Dict[str, Any]] = []
        self.temperature = temperature
        self.response_format = response_format
        self.response_schema = response_schema
        self.instructions = instructions
        self.tool_registry = tool_registry
        self.role_map = {"user": "user", "assistant": "assistant", "system": "system"}
        register_callback(self._invalidate_tools)
        self._invalidate_tools()

    def _invalidate_tools(self) -> None:
        self._convert_tools()

    def _convert_tools(self) -> None:
        self.together_tools = []
        all_tools = collect_tools(self.synaptic_tools, self.tool_registry)
        for tool in all_tools.values():
            self.together_tools.append({"type": "function", "function": tool.declaration})
        self.synaptic_tools = list(all_tools.values())

    def _request(self, prompt, role, images, audio, should_stream, kwargs) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model,
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
        response = self.client.chat.completions.create(**self._request(prompt, role, images, audio, False, kwargs))
        created = datetime.now().astimezone(timezone.utc)
        message = ""
        tool_calls: List[ToolCall] = []
        if response.choices:
            choice = response.choices[0]
            message = choice.message.content or ""
            tool_calls = parse_calls(choice.message)
        if self.response_format == ResponseFormat.JSON and not tool_calls:
            try:
                message = json.dumps(json.loads(message))
            except Exception:
                pass
        return ResponseMem(message=message, created=created, tool_calls=tool_calls)

    async def astream(
        self,
        prompt: Optional[str],
        role: str = "user",
        images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        abort=None,
        **kwargs,
    ) -> AsyncIterator[ResponseChunk]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[Any]] = asyncio.Queue()
        request = self._request(prompt, role, images, audio, True, kwargs)

        def producer() -> None:
            try:
                for chunk in self.client.chat.completions.create(**request):
                    if abort and abort.is_set():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=producer, daemon=True).start()
        accumulated = ""
        pending: Dict[int, Dict[str, str]] = {}

        while True:
            item = await q.get()
            if abort and abort.is_set():
                return
            if isinstance(item, Exception):
                raise item
            if item is None:
                break
            if not item.choices:
                continue
            delta = item.choices[0].delta
            if delta.content:
                accumulated += delta.content
                yield ResponseChunk(text=delta.content)
            for tc_delta in delta.tool_calls or []:
                current = pending.setdefault(tc_delta.index, {"name": "", "args": ""})
                if tc_delta.function.name:
                    current["name"] += tc_delta.function.name
                if tc_delta.function.arguments:
                    current["args"] += tc_delta.function.arguments

        for call in pending_calls(pending):
            yield ResponseChunk(text="", function_call=call)
        yield ResponseChunk(text=accumulated, is_final=True)
