import asyncio
import json
import threading
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import TOOL_REGISTRY, Tool, ToolCall, register_callback

load_dotenv()


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
    ):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model
        self.synaptic_tools = list(tools or [])
        self.openai_tools: List[Dict] = []
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.instructions = instructions
        self.role_map = {
            "user": "user",
            "assistant": "assistant",
            "system": "system",
        }

        register_callback(self._invalidate_tools)
        self._invalidate_tools()

    def _invalidate_tools(self):
        """Update OpenAI tools without mutating synaptic tools."""
        self._convert_tools()

    def _convert_tools(self) -> None:
        """Convert synaptic Tool objects + TOOL_REGISTRY to OpenAI function definitions."""
        self.openai_tools = []

        if self.response_format != ResponseFormat.NONE:
            return

        all_tools = {}
        for t in self.synaptic_tools or []:
            all_tools[t.name] = t
        for t_name, t in TOOL_REGISTRY.items():
            if t_name not in all_tools:
                all_tools[t_name] = t

        for t_name, t in all_tools.items():
            self.openai_tools.append({"type": "function", "function": t.declaration})

        # Fixed: was inside loop before, assigned once after
        self.synaptic_tools = list(all_tools.values())

    def to_messages(self) -> list[ChatCompletionMessageParam]:
        """Convert all memories to OpenAI ChatCompletion message objects."""
        messages: list[ChatCompletionMessageParam] = []

        if self.instructions:
            messages.append(
                ChatCompletionSystemMessageParam(
                    content=self.instructions, role="system"
                )
            )

        for memory in self.history.MemoryList:
            if isinstance(memory, ResponseMem) and memory.tool_calls:
                # Assistant turn with tool_calls — each call gets a unique ID by index
                messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=memory.message or "",
                        tool_calls=[
                            {
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.args) if tc.args else "{}",
                                },
                            }
                            for i, tc in enumerate(memory.tool_calls)
                        ],
                    )
                )

                # One tool result message per call, linked by ID
                tool_results = getattr(memory, "tool_results", None) or []
                for i, (tc, result) in enumerate(zip(memory.tool_calls, tool_results)):
                    content_str = json.dumps(result) if isinstance(result, dict) else str(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"call_{i}",
                            "name": tc.name,
                            "content": content_str,
                        }
                    )
            else:
                role = self.role_map.get(memory.role, "user")
                if role == "user":
                    messages.append(
                        ChatCompletionUserMessageParam(content=memory.message, role="user")
                    )
                elif role == "system":
                    messages.append(
                        ChatCompletionSystemMessageParam(content=memory.message, role="system")
                    )
                elif role == "assistant":
                    messages.append(
                        ChatCompletionAssistantMessageParam(content=memory.message, role="assistant")
                    )
                else:
                    messages.append(
                        ChatCompletionUserMessageParam(content=memory.message, role="user")
                    )

        return messages

    def invoke(self, prompt: str, role: str = "user", audio: Optional[List[str]] = None, **kwargs) -> ResponseMem:
        messages = self.to_messages()

        if role == "user":
            messages.append(ChatCompletionUserMessageParam(content=prompt, role="user"))
        elif role == "assistant":
            messages.append(
                ChatCompletionAssistantMessageParam(content=prompt, role="assistant")
            )
        elif role == "system":
            messages.append(
                ChatCompletionSystemMessageParam(content=prompt, role="system")
            )
        else:
            messages.append(ChatCompletionUserMessageParam(content=prompt, role="user"))

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **kwargs,
        }

        if self.openai_tools and self.response_format == ResponseFormat.NONE:
            params["tools"] = self.openai_tools
            params["tool_choice"] = "auto"

        if self.response_format == ResponseFormat.JSON:
            params["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**params)

        created = datetime.now().astimezone(timezone.utc)
        choice = response.choices[0]

        message = (
            choice.message.content if choice.message and choice.message.content else ""
        )

        tool_calls: List[ToolCall] = []
        if choice.message and getattr(choice.message, "tool_calls", None):
            for tc in choice.message.tool_calls:
                if hasattr(tc, "function"):
                    try:
                        args = (
                            json.loads(tc.function.arguments)
                            if tc.function.arguments
                            else {}
                        )
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(ToolCall(name=tc.function.name, args=args))

        return ResponseMem(message=message, created=created, tool_calls=tool_calls)

    async def astream(
        self, prompt: str, role: str = "user", audio: Optional[List[str]] = None, **kwargs
    ) -> AsyncIterator[ResponseChunk]:
        """
        Asynchronously stream response chunks from DeepSeek.

        Runs the sync streaming client in a background thread and pushes chunks
        into an asyncio.Queue. Tool call args are accumulated across deltas and
        emitted as ResponseChunk(function_call=...) after the text stream ends.
        """
        messages = self.to_messages()

        if role == "user":
            messages.append(ChatCompletionUserMessageParam(content=prompt, role="user"))
        elif role == "assistant":
            messages.append(
                ChatCompletionAssistantMessageParam(content=prompt, role="assistant")
            )
        else:
            messages.append(ChatCompletionUserMessageParam(content=prompt, role="user"))

        request_params: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        if self.openai_tools and self.response_format == ResponseFormat.NONE:
            request_params["tools"] = self.openai_tools
            request_params["tool_choice"] = "auto"

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[Any]] = asyncio.Queue()

        def producer():
            try:
                with self.client.chat.completions.create(**request_params) as stream:
                    for chunk in stream:
                        loop.call_soon_threadsafe(q.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=producer, daemon=True).start()

        accumulated_message = ""
        pending_tool_calls: Dict[int, Dict[str, str]] = {}

        while True:
            item = await q.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                break

            if not item.choices:
                continue

            delta = item.choices[0].delta

            if delta.content:
                accumulated_message += delta.content
                yield ResponseChunk(
                    text=delta.content, is_final=False, function_call=None
                )

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in pending_tool_calls:
                        pending_tool_calls[idx] = {"name": "", "args": ""}
                    if tc_delta.function.name:
                        pending_tool_calls[idx]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        pending_tool_calls[idx]["args"] += tc_delta.function.arguments

        for idx in sorted(pending_tool_calls):
            tc = pending_tool_calls[idx]
            try:
                args = json.loads(tc["args"]) if tc["args"] else {}
            except json.JSONDecodeError:
                args = {}
            tfc = ToolCall(name=tc["name"], args=args)
            yield ResponseChunk(text="", is_final=False, function_call=tfc)

        yield ResponseChunk(text=accumulated_message, is_final=True, function_call=None)
