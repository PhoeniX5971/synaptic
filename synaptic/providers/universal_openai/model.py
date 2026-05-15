import base64
import json
import mimetypes
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem, ToolCallArgsDelta
from ...core.tool import Tool, ToolCall, ToolRegistry, collect_tools, register_callback


class UniversalLLMAdapter(BaseModel):
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
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.client = OpenAI(api_key=api_key or "none", base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key or "none", base_url=base_url)
        self.model = model
        self.history = history
        self.bound_tools = list(tools or [])
        self.synaptic_tools = list(tools or [])
        self.openai_tools: List[Dict] = []
        self.temperature = temperature
        self.instructions = instructions
        self.response_format = response_format
        self.response_schema = response_schema
        self.tool_registry = tool_registry
        self.role_map = {"user": "user", "assistant": "assistant", "system": "system"}

        register_callback(self._invalidate_tools, tool_registry)
        self._invalidate_tools()

    def _invalidate_tools(self):
        self._convert_tools()

    def _convert_tools(self) -> None:
        self.openai_tools = []
        all_tools = collect_tools(self.bound_tools, self.tool_registry)
        for t_name, t in all_tools.items():
            self.openai_tools.append({"type": "function", "function": t.declaration})
        self.synaptic_tools = list(all_tools.values())

    def to_contents(self) -> List[Dict[str, Any]]:
        contents: List[Dict[str, Any]] = []
        if self.instructions:
            contents.append({"role": "system", "content": self.instructions})
        if self.history is None:
            return contents
        for memory in self.history.effective_mems():
            if isinstance(memory, ResponseMem) and memory.tool_calls:
                contents.append({
                    "role": "assistant", "content": memory.message or "",
                    "tool_calls": [
                        {"id": f"call_{i}", "type": "function",
                         "function": {"name": tc.name, "arguments": json.dumps(tc.args) if tc.args else "{}"}}
                        for i, tc in enumerate(memory.tool_calls)
                    ],
                })
                tool_results = getattr(memory, "tool_results", None) or []
                for i, (tc, result) in enumerate(zip(memory.tool_calls, tool_results)):
                    contents.append({
                        "role": "tool", "tool_call_id": f"call_{i}", "name": tc.name,
                        "content": json.dumps(result) if isinstance(result, dict) else str(result),
                    })
            else:
                contents.append({"role": self.role_map.get(memory.role, "user"), "content": memory.message})
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

    def _build_messages(self, prompt: Optional[str], role: str, audio: Optional[List[str]]):
        audio_blocks = self._audio_content_blocks(audio)
        messages = self.to_contents()
        if prompt is not None or audio_blocks:
            if audio_blocks:
                content = [{"type": "text", "text": prompt or ""}] + audio_blocks
            else:
                content = prompt or ""
            messages = messages + [{"role": role, "content": content}]
        return messages

    def invoke(self, prompt: Optional[str], role: str = "user", audio: Optional[List[str]] = None, **kwargs) -> ResponseMem:
        role = self.role_map.get(role, "user")
        messages = self._build_messages(prompt, role, audio)

        request_params: Dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": self.temperature,
            "max_tokens": kwargs.pop("max_tokens", 1024), **kwargs,
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
        role = self.role_map.get(role, "user")
        messages = self._build_messages(prompt, role, audio)

        request_params: Dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": self.temperature,
            "stream": True, "stream_options": {"include_usage": True}, **kwargs,
        }
        if self.openai_tools and self.response_format == ResponseFormat.NONE:
            request_params["tools"] = self.openai_tools
            request_params["tool_choice"] = "auto"

        accumulated_message = ""
        pending_tool_calls: Dict[int, Dict[str, str]] = {}
        u = None

        async for chunk in await self.async_client.chat.completions.create(**request_params):
            if abort and abort.is_set():
                return
            if getattr(chunk, "usage", None):
                u = chunk.usage
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
                        if pending_tool_calls[idx]["name"]:
                            yield ResponseChunk(text="", tool_call_delta=ToolCallArgsDelta(pending_tool_calls[idx]["name"], tc_delta.function.arguments, pending_tool_calls[idx]["args"]))

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
