import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import Tool, ToolCall, ToolRegistry, collect_tools, register_callback

load_dotenv()

_BASE_URL = "https://api.deepseek.com"


class DeepSeekAdapter(BaseModel):

    def __init__(
        self,
        model: str,
        history: History,
        api_key: str,
        response_format: ResponseFormat,
        response_schema: Any,
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.8,
        instructions: str = "",
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=_BASE_URL)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=_BASE_URL)
        self.model = model
        self.synaptic_tools = list(tools or [])
        self.openai_tools: List[Dict] = []
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.instructions = instructions
        self.tool_registry = tool_registry
        self.role_map = {"user": "user", "assistant": "assistant", "system": "system"}

        register_callback(self._invalidate_tools)
        self._invalidate_tools()

    def _invalidate_tools(self):
        self._convert_tools()

    def _convert_tools(self) -> None:
        self.openai_tools = []
        if self.response_format != ResponseFormat.NONE:
            return
        all_tools = collect_tools(self.synaptic_tools, self.tool_registry)
        for t_name, t in all_tools.items():
            self.openai_tools.append({"type": "function", "function": t.declaration})
        self.synaptic_tools = list(all_tools.values())

    def to_messages(self) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        if self.instructions:
            messages.append(ChatCompletionSystemMessageParam(content=self.instructions, role="system"))
        for memory in self.history.MemoryList:
            if isinstance(memory, ResponseMem) and memory.tool_calls:
                messages.append(ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=memory.message or "",
                    tool_calls=[
                        {"id": f"call_{i}", "type": "function",
                         "function": {"name": tc.name, "arguments": json.dumps(tc.args) if tc.args else "{}"}}
                        for i, tc in enumerate(memory.tool_calls)
                    ],
                ))
                tool_results = getattr(memory, "tool_results", None) or []
                for i, (tc, result) in enumerate(zip(memory.tool_calls, tool_results)):
                    messages.append({
                        "role": "tool", "tool_call_id": f"call_{i}", "name": tc.name,
                        "content": json.dumps(result) if isinstance(result, dict) else str(result),
                    })
            else:
                role = self.role_map.get(memory.role, "user")
                if role == "assistant":
                    messages.append(ChatCompletionAssistantMessageParam(content=memory.message, role="assistant"))
                elif role == "system":
                    messages.append(ChatCompletionSystemMessageParam(content=memory.message, role="system"))
                else:
                    messages.append(ChatCompletionUserMessageParam(content=memory.message, role="user"))
        return messages

    def _append_prompt(self, messages, prompt: Optional[str], role: str):
        if prompt is None:
            return
        if role == "assistant":
            messages.append(ChatCompletionAssistantMessageParam(content=prompt, role="assistant"))
        elif role == "system":
            messages.append(ChatCompletionSystemMessageParam(content=prompt, role="system"))
        else:
            messages.append(ChatCompletionUserMessageParam(content=prompt, role="user"))

    def invoke(self, prompt: Optional[str], role: str = "user", audio: Optional[List[str]] = None, **kwargs) -> ResponseMem:
        messages = self.to_messages()
        self._append_prompt(messages, prompt, role)

        params: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature, **kwargs}
        if self.openai_tools and self.response_format == ResponseFormat.NONE:
            params["tools"] = self.openai_tools
            params["tool_choice"] = "auto"
        if self.response_format == ResponseFormat.JSON:
            params["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**params)
        created = datetime.now().astimezone(timezone.utc)
        choice = response.choices[0]
        message = choice.message.content if choice.message and choice.message.content else ""
        tool_calls: List[ToolCall] = []
        if choice.message and getattr(choice.message, "tool_calls", None):
            for tc in choice.message.tool_calls:
                if hasattr(tc, "function"):
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(ToolCall(name=tc.function.name, args=args))
        u = getattr(response, "usage", None)
        return ResponseMem(
            message=message, created=created, tool_calls=tool_calls,
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
        )

    async def astream(
        self, prompt: Optional[str], role: str = "user", audio: Optional[List[str]] = None,
        abort=None, **kwargs
    ) -> AsyncIterator[ResponseChunk]:
        messages = self.to_messages()
        self._append_prompt(messages, prompt, role)

        request_params: Dict[str, Any] = {
            "model": self.model, "temperature": self.temperature, "messages": messages,
            "stream": True, **kwargs,
        }
        if self.openai_tools and self.response_format == ResponseFormat.NONE:
            request_params["tools"] = self.openai_tools
            request_params["tool_choice"] = "auto"

        request_params["stream_options"] = {"include_usage": True}
        accumulated_message = ""
        pending_tool_calls: Dict[int, Dict[str, str]] = {}

        async with self.async_client.chat.completions.stream(**request_params) as stream:
            async for chunk in stream:
                if abort and abort.is_set():
                    return
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    accumulated_message += delta.content
                    yield ResponseChunk(text=delta.content, is_final=False, function_call=None)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {"name": "", "args": ""}
                        if tc_delta.function.name:
                            pending_tool_calls[idx]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            pending_tool_calls[idx]["args"] += tc_delta.function.arguments
            u = getattr(stream, "usage", None)

        for idx in sorted(pending_tool_calls):
            tc = pending_tool_calls[idx]
            try:
                args = json.loads(tc["args"]) if tc["args"] else {}
            except json.JSONDecodeError:
                args = {}
            yield ResponseChunk(text="", is_final=False, function_call=ToolCall(name=tc["name"], args=args))

        yield ResponseChunk(text=accumulated_message, is_final=True, function_call=None,
                            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                            output_tokens=getattr(u, "completion_tokens", 0) or 0)
