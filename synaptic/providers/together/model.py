import asyncio
import base64
import json
import mimetypes
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from dotenv import load_dotenv
from together import Together

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import TOOL_REGISTRY, Tool, ToolCall, register_callback

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
        stream: bool = False,
    ):
        self.client = Together(api_key=api_key)
        self.model = model
        self.synaptic_tools = list(tools or [])
        self.together_tools: List[Dict] = []
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.instructions = instructions
        self.stream = stream

        self.role_map = {
            "user": "user",
            "assistant": "assistant",
            "system": "system",
        }

        register_callback(self._invalidate_tools)
        self._invalidate_tools()

    def _invalidate_tools(self):
        """Update Together tools when registry changes."""
        self._convert_tools()

    def _convert_tools(self) -> None:
        """Convert synaptic tools + TOOL_REGISTRY to Together/OpenAI tool format."""
        self.together_tools = []

        all_tools = {}
        for t in self.synaptic_tools or []:
            all_tools[t.name] = t
        for t_name, t in TOOL_REGISTRY.items():
            if t_name not in all_tools:
                all_tools[t_name] = t

        for t_name, t in all_tools.items():
            self.together_tools.append({"type": "function", "function": t.declaration})

        self.synaptic_tools = list(all_tools.values())

    def to_messages(self) -> List[Dict]:
        """Convert all memories to Together/OpenAI chat message format."""
        messages: List[Dict] = []

        if self.history is None:
            return messages

        for memory in self.history.MemoryList:
            if isinstance(memory, ResponseMem) and memory.tool_calls:
                # Assistant turn with tool_calls — each call gets a unique ID by index
                messages.append(
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
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"call_{i}",
                            "name": tc.name,
                            "content": content_str,
                        }
                    )
            else:
                messages.append(
                    {
                        "role": self.role_map.get(memory.role, "user"),
                        "content": memory.message,
                    }
                )

        return messages

    def _image_to_url(self, img: str) -> str:
        """Return a data URI for local files, or pass through URLs as-is."""
        p = Path(img)
        if p.exists():
            mime, _ = mimetypes.guess_type(img)
            mime = mime or "application/octet-stream"
            data = base64.b64encode(p.read_bytes()).decode()
            return f"data:{mime};base64,{data}"
        return img

    def _build_content(self, prompt: str, images: Optional[List[str]]) -> Any:
        """Return plain string content or a multipart content list when images are given."""
        if not images:
            return prompt
        parts: List[Dict] = [{"type": "text", "text": prompt}]
        for img in images:
            parts.append({"type": "image_url", "image_url": {"url": self._image_to_url(img)}})
        return parts

    def invoke(self, prompt: str, role: str = "user", images: Optional[List[str]] = None, **kwargs) -> ResponseMem:
        role = self.role_map.get(role, "user")

        messages: List[Dict] = []

        system_message = self.instructions or ""
        if self.response_format == ResponseFormat.JSON:
            system_message += "\nYou must respond ONLY with valid JSON. Do not include explanations or markdown."

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.extend(self.to_messages())
        messages.append({"role": role, "content": self._build_content(prompt, images)})

        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if self.together_tools and self.response_format == ResponseFormat.NONE:
            request_params["tools"] = self.together_tools
            request_params["tool_choice"] = "auto"

        if self.response_format == ResponseFormat.JSON:
            request_params["response_format"] = {"type": "json_object"}

        request_params.update(kwargs)

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

        if self.response_format == ResponseFormat.JSON and not tool_calls:
            try:
                parsed = json.loads(message)
                message = json.dumps(parsed)
            except Exception:
                pass

        return ResponseMem(
            message=message,
            created=created,
            tool_calls=tool_calls,
        )

    async def astream(
        self, prompt: str, role: str = "user", images: Optional[List[str]] = None, **kwargs
    ) -> AsyncIterator[ResponseChunk]:
        """
        Asynchronously stream response chunks from Together AI.

        Runs the sync streaming client in a background thread and pushes chunks
        into an asyncio.Queue. Tool call args are accumulated across deltas and
        emitted after the text stream ends.
        """
        role = self.role_map.get(role, "user")

        messages: List[Dict] = []

        system_message = self.instructions or ""
        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.extend(self.to_messages())
        messages.append({"role": role, "content": self._build_content(prompt, images)})

        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            **kwargs,
        }

        if self.together_tools and self.response_format == ResponseFormat.NONE:
            request_params["tools"] = self.together_tools
            request_params["tool_choice"] = "auto"

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[Any]] = asyncio.Queue()

        def producer():
            try:
                for chunk in self.client.chat.completions.create(**request_params):
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
