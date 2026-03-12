import asyncio
import json
import threading
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import anthropic
from dotenv import load_dotenv

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import TOOL_REGISTRY, Tool, ToolCall, register_callback

load_dotenv()

_STRUCTURED_OUTPUT_TOOL = "__structured_output__"


class ClaudeAdapter(BaseModel):
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
        max_tokens: int = 1024,
    ):
        self.client = anthropic.Anthropic(api_key=api_key or None)
        self.model = model
        self.synaptic_tools = list(tools or [])
        self.claude_tools: List[Dict[str, Any]] = []
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.instructions = instructions
        self.max_tokens = max_tokens

        register_callback(self._invalidate_tools)
        self._invalidate_tools()

    def _invalidate_tools(self) -> None:
        self._convert_tools()

    def _convert_tools(self) -> None:
        """Convert synaptic Tool objects + TOOL_REGISTRY to Anthropic tool dicts."""
        all_tools: Dict[str, Tool] = {}
        for t in self.synaptic_tools or []:
            all_tools[t.name] = t
        for t_name, t in TOOL_REGISTRY.items():
            if t_name not in all_tools:
                all_tools[t_name] = t

        self.claude_tools = [
            {
                "name": t.declaration["name"],
                "description": t.declaration.get("description", ""),
                "input_schema": t.declaration.get("parameters", {"type": "object", "properties": {}}),
            }
            for t in all_tools.values()
        ]
        self.synaptic_tools = list(all_tools.values())

    def _make_tool_id(self, turn_index: int, call_index: int) -> str:
        """Generate a deterministic tool_use id for history reconstruction."""
        return f"toolu_{turn_index:03d}_{call_index:02d}"

    def to_contents(self) -> List[Dict[str, Any]]:
        """Convert conversation history to Anthropic Messages API format."""
        messages: List[Dict[str, Any]] = []

        if self.history is None:
            return messages

        for turn_idx, memory in enumerate(self.history.MemoryList):
            if isinstance(memory, ResponseMem) and memory.tool_calls:
                # Assistant turn: text + tool_use blocks
                assistant_content: List[Dict[str, Any]] = []
                if memory.message:
                    assistant_content.append({"type": "text", "text": memory.message})
                for call_idx, tc in enumerate(memory.tool_calls):
                    assistant_content.append({
                        "type": "tool_use",
                        "id": self._make_tool_id(turn_idx, call_idx),
                        "name": tc.name,
                        "input": tc.args,
                    })
                messages.append({"role": "assistant", "content": assistant_content})

                # User turn: tool_result blocks
                tool_results = getattr(memory, "tool_results", None) or []
                result_content: List[Dict[str, Any]] = []
                for call_idx, tc in enumerate(memory.tool_calls):
                    result = tool_results[call_idx] if call_idx < len(tool_results) else {}
                    if isinstance(result, dict):
                        resp = result.get("result", result.get("error", ""))
                    else:
                        resp = str(result)
                    result_content.append({
                        "type": "tool_result",
                        "tool_use_id": self._make_tool_id(turn_idx, call_idx),
                        "content": str(resp),
                    })
                if result_content:
                    messages.append({"role": "user", "content": result_content})
            else:
                role = "assistant" if memory.role == "assistant" else "user"
                messages.append({"role": role, "content": memory.message})

        return messages

    def _build_request(self, prompt: str, role: str, stream: bool = False, **kwargs) -> Dict[str, Any]:
        """Build the Anthropic messages.create request params."""
        messages = self.to_contents()
        messages.append({"role": role, "content": prompt})

        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": self.temperature,
            "messages": messages,
        }

        if self.instructions:
            params["system"] = self.instructions

        if self.response_format == ResponseFormat.JSON and self.response_schema is not None:
            # Use tool-forcing for reliable structured/Pydantic JSON output
            schema = (
                self.response_schema.model_json_schema()
                if hasattr(self.response_schema, "model_json_schema")
                else self.response_schema
            )
            params["tools"] = [{
                "name": _STRUCTURED_OUTPUT_TOOL,
                "description": "Return the structured response conforming to the required schema.",
                "input_schema": schema,
            }]
            params["tool_choice"] = {"type": "tool", "name": _STRUCTURED_OUTPUT_TOOL}
        elif self.claude_tools and self.response_format == ResponseFormat.NONE:
            params["tools"] = self.claude_tools

        return params

    def invoke(
        self,
        prompt: str,
        role: str = "user",
        **kwargs,
    ) -> ResponseMem:
        role = "assistant" if role == "assistant" else "user"
        params = self._build_request(prompt, role, **kwargs)

        response = self.client.messages.create(**params)

        created = datetime.now().astimezone(timezone.utc)
        message = ""
        tool_calls: List[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                message += block.text
            elif block.type == "tool_use":
                if block.name == _STRUCTURED_OUTPUT_TOOL:
                    # Structured output: serialize the input dict as JSON string
                    message = json.dumps(block.input)
                else:
                    tool_calls.append(ToolCall(name=block.name, args=block.input or {}))

        return ResponseMem(message=message, created=created, tool_calls=tool_calls)

    async def astream(
        self,
        prompt: str,
        role: str = "user",
        **kwargs,
    ) -> AsyncIterator[ResponseChunk]:
        """
        Asynchronously stream response chunks from Claude.

        Runs the sync Anthropic MessageStream in a background thread and
        bridges events into an asyncio.Queue. Yields ResponseChunk objects
        with partial text and completed tool_call events.
        """
        role = "assistant" if role == "assistant" else "user"
        params = self._build_request(prompt, role, **kwargs)

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Any] = asyncio.Queue()

        def producer():
            try:
                with self.client.messages.stream(**params) as stream:
                    for event in stream:
                        loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=producer, daemon=True).start()

        accumulated_message = ""
        tool_calls: List[ToolCall] = []

        # Per-block accumulation state
        active_tool_name: Optional[str] = None
        active_tool_json: str = ""
        is_structured_output_block = False

        while True:
            item = await q.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                break

            event_type = getattr(item, "type", None)

            if event_type == "content_block_start":
                block = getattr(item, "content_block", None)
                if block and getattr(block, "type", None) == "tool_use":
                    active_tool_name = block.name
                    active_tool_json = ""
                    is_structured_output_block = (block.name == _STRUCTURED_OUTPUT_TOOL)

            elif event_type == "content_block_delta":
                delta = getattr(item, "delta", None)
                if delta is None:
                    continue

                delta_type = getattr(delta, "type", None)

                if delta_type == "text_delta":
                    piece = delta.text
                    accumulated_message += piece
                    yield ResponseChunk(text=piece, is_final=False, function_call=None)

                elif delta_type == "input_json_delta":
                    active_tool_json += delta.partial_json

            elif event_type == "content_block_stop":
                if active_tool_name is not None:
                    try:
                        args = json.loads(active_tool_json) if active_tool_json else {}
                    except json.JSONDecodeError:
                        args = {}

                    if is_structured_output_block:
                        # Emit structured output as text
                        json_str = json.dumps(args)
                        accumulated_message = json_str
                        yield ResponseChunk(text=json_str, is_final=False, function_call=None)
                    else:
                        tfc = ToolCall(name=active_tool_name, args=args)
                        tool_calls.append(tfc)
                        yield ResponseChunk(text="", is_final=False, function_call=tfc)

                    active_tool_name = None
                    active_tool_json = ""
                    is_structured_output_block = False

        yield ResponseChunk(text=accumulated_message, is_final=True, function_call=None)
