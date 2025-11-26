import json
from datetime import datetime, timezone
from typing import Any, List, Optional, Dict
from openai import OpenAI

from ...core.base import BaseModel, History, ResponseFormat, ResponseMem
from ...core.tool import TOOL_REGISTRY, ToolCall, register_callback


class OpenAIAdapter(BaseModel):
    def __init__(
        self,
        model: str,
        history: History,
        api_key: str,
        response_format: ResponseFormat,
        response_schema: Any,
        temperature: float = 0.8,
        tools: list | None = None,
        instructions: str = "",
        **kwargs,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.synaptic_tools = tools or []
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
        """Update OpenAI tools when registry changes"""
        self._convert_tools()

    def _convert_tools(self) -> None:
        """Convert synaptic tools + TOOL_REGISTRY to OpenAI tool format"""
        self.openai_tools = []

        # Deduplicate tools
        all_tools = {}
        for t in self.synaptic_tools or []:
            all_tools[t.name] = t
        for t_name, t in TOOL_REGISTRY.items():
            if t_name not in all_tools:
                all_tools[t_name] = t

        # Convert to OpenAI tool format
        for t_name, t in all_tools.items():
            self.openai_tools.append({"type": "function", "function": t.declaration})

        self.synaptic_tools = list(all_tools.values())

    def to_contents(self) -> List[Dict[str, Any]]:
        """Convert memory list to OpenAI-style messages with proper tool messages."""
        contents = []

        for memory in self.history.MemoryList:
            # Base message
            message_content = memory.message
            if hasattr(memory, "created"):
                message_content += f" (Created at: {memory.created})"

            message: Dict[str, Any] = {
                "role": self.role_map.get(memory.role, "user"),
                "content": message_content,
            }

            # Add tool calls for assistant responses
            if isinstance(memory, ResponseMem) and getattr(memory, "tool_calls", None):
                message["tool_calls"] = [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.args) if call.args else "{}",
                        },
                    }
                    for i, call in enumerate(memory.tool_calls)
                ]

            contents.append(message)

            # Add separate tool messages for each tool result
            if isinstance(memory, ResponseMem) and getattr(
                memory, "tool_results", None
            ):
                for i, result in enumerate(memory.tool_results):
                    # Match tool_call_id by index
                    tool_message = {
                        "role": "tool",
                        "name": (
                            memory.tool_calls[i].name
                            if memory.tool_calls
                            else f"tool_{i}"
                        ),
                        "content": (
                            json.dumps(result)
                            if isinstance(result, (dict, list))
                            else str(result)
                        ),
                        "tool_call_id": f"call_{i}",
                    }
                    contents.append(tool_message)

        # Add system instructions if provided
        if self.instructions:
            contents.insert(0, {"role": "system", "content": self.instructions})

        return contents

    def invoke(self, prompt: str, role: str = "user", **kwargs) -> ResponseMem:
        """Invoke OpenAI model with modern API using to_contents style"""
        role = self.role_map.get(role, "user")

        messages = self.to_contents() + [{"role": role, "content": prompt}]

        request_params = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
        }

        # Handle response format
        if self.response_format == ResponseFormat.JSON:
            request_params["response_format"] = {"type": "json_object"}
        elif hasattr(self.response_schema, "model_json_schema"):
            request_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.response_schema.__name__,
                    "schema": self.response_schema.model_json_schema(),
                },
            }

        if self.openai_tools and self.response_format == ResponseFormat.NONE:
            request_params["tools"] = self.openai_tools
            request_params["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**request_params)
        except Exception:
            response = self.client.chat.completions.create(**request_params)

        created = datetime.now().astimezone(timezone.utc)
        message = ""
        tool_calls: List[ToolCall] = []

        if response.choices:
            choice = response.choices[0]
            message_content = choice.message.content or ""
            message = message_content

            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    if hasattr(tool_call, "function"):
                        try:
                            args = (
                                json.loads(tool_call.function.arguments)
                                if tool_call.function.arguments
                                else {}
                            )
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append(
                            ToolCall(name=tool_call.function.name, args=args)
                        )

        return ResponseMem(message=message, created=created, tool_calls=tool_calls)
