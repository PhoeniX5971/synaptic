from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem, ToolCallArgsDelta
from ...core.tool import Tool, ToolCall, ToolRegistry, collect_tools, register_callback
from .helpers import add_prompt, history_messages, parse_tool_calls, stream_tool_calls


class OpenAIAdapter(BaseModel):
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
        **kwargs,
    ):
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.history = history
        self.bound_tools = list(tools or [])
        self.synaptic_tools = list(tools or [])
        self.openai_tools: List[Dict[str, Any]] = []
        self.temperature = temperature
        self.response_format = response_format
        self.response_schema = response_schema
        self.instructions = instructions
        self.tool_registry = tool_registry
        self.role_map = {"user": "user", "assistant": "assistant", "system": "system"}
        register_callback(self._invalidate_tools, tool_registry)
        self._invalidate_tools()

    def _invalidate_tools(self) -> None:
        self._convert_tools()

    def _convert_tools(self) -> None:
        self.openai_tools = []
        all_tools = collect_tools(self.bound_tools, self.tool_registry)
        for tool in all_tools.values():
            self.openai_tools.append({"type": "function", "function": tool.declaration})
        self.synaptic_tools = list(all_tools.values())

    def _messages(self, prompt: Optional[str], role: str, audio: Optional[List[str]]) -> List[Dict[str, Any]]:
        messages = history_messages(self.history, self.instructions, self.role_map)
        add_prompt(messages, prompt, self.role_map.get(role, "user"), audio)
        return messages

    def _request(self, messages: List[Dict[str, Any]], streaming: bool, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            **kwargs,
        }
        if not streaming:
            params["max_tokens"] = params.pop("max_tokens", 1024)
        else:
            params["stream"] = True
        if self.response_format == ResponseFormat.JSON:
            if hasattr(self.response_schema, "model_json_schema"):
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.response_schema.__name__,
                        "schema": self.response_schema.model_json_schema(),
                    },
                }
            else:
                params["response_format"] = {"type": "json_object"}
        elif self.openai_tools:
            params["tools"] = self.openai_tools
            params["tool_choice"] = "auto"
        return params

    def invoke(
        self,
        prompt: Optional[str],
        role: str = "user",
        audio: Optional[List[str]] = None,
        **kwargs,
    ) -> ResponseMem:
        params = self._request(self._messages(prompt, role, audio), False, kwargs)
        response = self.client.chat.completions.create(**params)
        created = datetime.now().astimezone(timezone.utc)
        message = ""
        tool_calls: List[ToolCall] = []
        if response.choices:
            choice = response.choices[0]
            message = choice.message.content or ""
            tool_calls = parse_tool_calls(choice.message)
        u = getattr(response, "usage", None)
        return ResponseMem(
            message=message, created=created, tool_calls=tool_calls,
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
        )

    async def astream(
        self,
        prompt: Optional[str],
        role: str = "user",
        audio: Optional[List[str]] = None,
        abort=None,
        **kwargs,
    ) -> AsyncIterator[ResponseChunk]:
        params = self._request(self._messages(prompt, role, audio), True, kwargs)
        params["stream_options"] = {"include_usage": True}
        accumulated = ""
        pending: Dict[int, Dict[str, str]] = {}

        async with self.async_client.chat.completions.stream(**params) as stream:
            async for chunk in stream:
                if abort and abort.is_set():
                    return
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    accumulated += delta.content
                    yield ResponseChunk(text=delta.content)
                for tc_delta in delta.tool_calls or []:
                    current = pending.setdefault(tc_delta.index, {"name": "", "args": ""})
                    if tc_delta.function.name:
                        current["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        current["args"] += tc_delta.function.arguments
                        if current["name"]:
                            yield ResponseChunk(text="", tool_call_delta=ToolCallArgsDelta(current["name"], tc_delta.function.arguments, current["args"]))
            u = getattr(stream, "usage", None)

        for call in stream_tool_calls(pending):
            yield ResponseChunk(text="", function_call=call)
        yield ResponseChunk(text=accumulated, is_final=True,
                            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                            output_tokens=getattr(u, "completion_tokens", 0) or 0)
