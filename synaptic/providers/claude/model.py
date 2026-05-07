import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import anthropic
from dotenv import load_dotenv

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import Tool, ToolCall, ToolRegistry, collect_tools, register_callback
from .helpers import history_messages, safe_json, schema_for

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
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.client = anthropic.Anthropic(api_key=api_key or None)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key or None)
        self.model = model
        self.synaptic_tools = list(tools or [])
        self.claude_tools: List[Dict[str, Any]] = []
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.instructions = instructions
        self.max_tokens = max_tokens
        self.tool_registry = tool_registry

        register_callback(self._invalidate_tools)
        self._invalidate_tools()

    def _invalidate_tools(self) -> None:
        self._convert_tools()

    def _convert_tools(self) -> None:
        """Convert synaptic Tool objects + TOOL_REGISTRY to Anthropic tool dicts."""
        all_tools = collect_tools(self.synaptic_tools, self.tool_registry)

        self.claude_tools = [
            {
                "name": t.declaration["name"],
                "description": t.declaration.get("description", ""),
                "input_schema": t.declaration.get("parameters", {"type": "object", "properties": {}}),
            }
            for t in all_tools.values()
        ]
        self.synaptic_tools = list(all_tools.values())

    def to_contents(self) -> List[Dict[str, Any]]:
        return history_messages(self.history)

    def _build_request(self, prompt: Optional[str], role: str, **kwargs) -> Dict[str, Any]:
        """Build the Anthropic messages.create request params."""
        messages = self.to_contents()
        if prompt is not None:
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
            schema = schema_for(self.response_schema)
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
        prompt: Optional[str],
        role: str = "user",
        audio: Optional[List[str]] = None,
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
                    message = json.dumps(block.input)
                else:
                    tool_calls.append(ToolCall(name=block.name, args=block.input or {}))

        usage = getattr(response, "usage", None)
        return ResponseMem(
            message=message, created=created, tool_calls=tool_calls,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    async def astream(
        self,
        prompt: Optional[str],
        role: str = "user",
        audio: Optional[List[str]] = None,
        abort=None,
        **kwargs,
    ) -> AsyncIterator[ResponseChunk]:
        role = "assistant" if role == "assistant" else "user"
        params = self._build_request(prompt, role, **kwargs)

        accumulated_message = ""
        tool_calls: List[ToolCall] = []
        active_tool_name: Optional[str] = None
        active_tool_json: str = ""
        is_structured_output_block = False
        input_tokens = 0
        output_tokens = 0

        async with self.async_client.messages.stream(**params) as stream:
            async for event in stream:
                if abort and abort.is_set():
                    return

                event_type = getattr(event, "type", None)

                if event_type == "message_start":
                    u = getattr(getattr(event, "message", None), "usage", None)
                    if u:
                        input_tokens = getattr(u, "input_tokens", 0) or 0
                elif event_type == "message_delta":
                    u = getattr(event, "usage", None)
                    if u:
                        output_tokens = getattr(u, "output_tokens", 0) or 0
                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        active_tool_name = block.name
                        active_tool_json = ""
                        is_structured_output_block = (block.name == _STRUCTURED_OUTPUT_TOOL)

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
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
                        args = safe_json(active_tool_json)
                        if is_structured_output_block:
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

        yield ResponseChunk(text=accumulated_message, is_final=True, function_call=None,
                            input_tokens=input_tokens, output_tokens=output_tokens)
