import asyncio
import base64
import json
import mimetypes
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import OpenAI

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import TOOL_REGISTRY, Tool, ToolCall, register_callback


class UniversalLLMAdapter(BaseModel):
    """
    Adapter for any OpenAI-compatible API (vLLM, Ollama, local servers, etc.).

    Supports:
    - invoke() and astream()
    - tool/function calling
    - chat history
    - optional JSON output mode
    - system instructions
    """

    def __init__(
        self,
        model: str,
        history: History | None,
        base_url: str,
        api_key: Optional[str],
        response_format: ResponseFormat,
        response_schema: Any = None,
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.8,
        instructions: str = "",
    ):
        self.client = OpenAI(
            api_key=api_key or "none",
            base_url=base_url,
        )
        self.model = model
        self.history = history
        self.synaptic_tools = list(tools or [])
        self.openai_tools: List[Dict] = []
        self.temperature = temperature
        self.instructions = instructions
        self.response_format = response_format
        self.response_schema = response_schema

        self.role_map = {
            "user": "user",
            "assistant": "assistant",
            "system": "system",
        }

        register_callback(self._invalidate_tools)
        self._invalidate_tools()

    def _invalidate_tools(self):
        """Update tools when registry changes."""
        self._convert_tools()

    def _convert_tools(self) -> None:
        """Convert synaptic tools + TOOL_REGISTRY to OpenAI tool format."""
        self.openai_tools = []

        all_tools = {}
        for t in self.synaptic_tools or []:
            all_tools[t.name] = t
        for t_name, t in TOOL_REGISTRY.items():
            if t_name not in all_tools:
                all_tools[t_name] = t

        for t_name, t in all_tools.items():
            self.openai_tools.append({"type": "function", "function": t.declaration})

        self.synaptic_tools = list(all_tools.values())

    def to_contents(self) -> List[Dict[str, Any]]:
        """Convert memory list to OpenAI-style messages."""
        contents: List[Dict[str, Any]] = []

        if self.instructions:
            contents.append({"role": "system", "content": self.instructions})

        if self.history is None:
            return contents

        for memory in self.history.MemoryList:
            if isinstance(memory, ResponseMem) and memory.tool_calls:
                # Assistant turn with tool_calls — each call gets a unique ID by index
                contents.append(
                    {
                        "role": "assistant",
                        "content": memory.message or "",
                        "tool_calls": [
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
                    }
                )

                # One tool result message per call, linked by ID
                tool_results = getattr(memory, "tool_results", None) or []
                for i, (tc, result) in enumerate(zip(memory.tool_calls, tool_results)):
                    content_str = json.dumps(result) if isinstance(result, dict) else str(result)
                    contents.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"call_{i}",
                            "name": tc.name,
                            "content": content_str,
                        }
                    )
            else:
                contents.append(
                    {
                        "role": self.role_map.get(memory.role, "user"),
                        "content": memory.message,
                    }
                )

        return contents

    def _audio_content_blocks(self, audio: Optional[List[str]]) -> List[Dict]:
        blocks = []
        for src in (audio or []):
            p = Path(src)
            if p.exists():
                data = p.read_bytes()
                mime, _ = mimetypes.guess_type(src)
                mime = mime or "audio/mpeg"
            else:
                with urllib.request.urlopen(src) as r:
                    data = r.read()
                    mime = r.headers.get_content_type() or "audio/mpeg"
            fmt = mime.split("/")[-1].replace("mpeg", "mp3")
            blocks.append({"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": fmt}})
        return blocks

    def invoke(self, prompt: str, role: str = "user", audio: Optional[List[str]] = None, **kwargs) -> ResponseMem:
        role = self.role_map.get(role, "user")

        audio_blocks = self._audio_content_blocks(audio)
        content = [{"type": "text", "text": prompt}] + audio_blocks if audio_blocks else prompt
        messages = self.to_contents() + [{"role": role, "content": content}]

        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            **kwargs,
        }

        if self.openai_tools and self.response_format == ResponseFormat.NONE:
            request_params["tools"] = self.openai_tools
            request_params["tool_choice"] = "auto"

        if self.response_format == ResponseFormat.JSON:
            request_params["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**request_params)

        created = datetime.now().astimezone(timezone.utc)
        message = ""
        tool_calls: List[ToolCall] = []

        if response.choices:
            choice = response.choices[0]
            message = choice.message.content or ""

            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
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
        Asynchronously stream response chunks from any OpenAI-compatible endpoint.

        Runs the sync streaming client in a background thread and pushes chunks
        into an asyncio.Queue. Tool call args are accumulated across deltas and
        emitted after the text stream ends.
        """
        role = self.role_map.get(role, "user")

        audio_blocks = self._audio_content_blocks(audio)
        content = [{"type": "text", "text": prompt}] + audio_blocks if audio_blocks else prompt
        messages = self.to_contents() + [{"role": role, "content": content}]

        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
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
