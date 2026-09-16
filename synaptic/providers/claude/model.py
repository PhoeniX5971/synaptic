from ._stream import ClaudeStream
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import anthropic

from ...core.base import BaseModel, History, ResponseFormat, ResponseMem
from ...core.tool import Tool, ToolCall, ToolRegistry, collect_tools, register_callback
from .helpers import history_messages, schema_for


_STRUCTURED_OUTPUT_TOOL = "__structured_output__"


from ...core.capabilities import Capabilities
from .._media import claude_content


class ClaudeAdapter(ClaudeStream, BaseModel):
    capabilities = Capabilities(images=True)

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
        self.bound_tools = list(tools or [])
        self.synaptic_tools = list(tools or [])
        self.claude_tools: List[Dict[str, Any]] = []
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.instructions = instructions
        self.max_tokens = max_tokens
        self.tool_registry = tool_registry

        self._subscription = register_callback(self._invalidate_tools, tool_registry)
        self._invalidate_tools()

    def _invalidate_tools(self) -> None:
        self._convert_tools()

    def _convert_tools(self) -> None:
        all_tools = collect_tools(self.bound_tools, self.tool_registry)

        self.claude_tools = [
            {
                "name": t.declaration["name"],
                "description": t.declaration.get("description", ""),
                "input_schema": t.declaration.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            for t in all_tools.values()
        ]
        self.synaptic_tools = list(all_tools.values())

    def to_contents(self) -> List[Dict[str, Any]]:
        return history_messages(self.history)

    def _build_request(
        self, prompt: Optional[str], role: str, images=None, **kwargs
    ) -> Dict[str, Any]:
        messages = self.to_contents()
        if prompt is not None or images:
            messages.append({"role": role, "content": claude_content(prompt, images)})

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
            params["tools"] = [
                {
                    "name": _STRUCTURED_OUTPUT_TOOL,
                    "description": "Return the structured response conforming to the required schema.",
                    "input_schema": schema,
                }
            ]
            params["tool_choice"] = {"type": "tool", "name": _STRUCTURED_OUTPUT_TOOL}
        elif self.claude_tools and self.response_format == ResponseFormat.NONE:
            params["tools"] = self.claude_tools

        return {**params, **kwargs}

    def invoke(
        self,
        prompt: Optional[str],
        role: str = "user",
        audio: Optional[List[str]] = None,
        images=None,
        **kwargs,
    ) -> ResponseMem:
        self.capabilities.check("invoke", images=images, audio=audio)
        role = "assistant" if role == "assistant" else "user"
        params = self._build_request(prompt, role, images=images, **kwargs)

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
                    tool_calls.append(
                        ToolCall(name=block.name, args=block.input or {}, id=block.id)
                    )

        usage = getattr(response, "usage", None)
        return ResponseMem(
            message=message,
            created=created,
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            finish_reason=getattr(response, "stop_reason", None),
            extensions={"response_id": getattr(response, "id", None)},
        )
