import json
from datetime import datetime, timezone
from typing import Any, List, Optional, Dict

from pydantic import BaseModel as PBM, create_model, ConfigDict

import google.genai as genai
from dotenv import load_dotenv
from google.genai import types

from ...core.base import BaseModel, History, ResponseFormat, ResponseMem
from ...core.tool import TOOL_REGISTRY, ToolCall, register_callback

load_dotenv()


class PToolCall(PBM):
    name: str
    args: Optional[Dict[str, Any]] = None


class GeminiAdapter(BaseModel):
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
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.synaptic_tools = tools or []
        self.gemini_tools: List[types.Tool] = []
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        register_callback(self._invalidate_tools)
        self._invalidate_tools()
        self.instructions = instructions
        self.role_map = {
            "user": "user",
            "assistant": "model",
            "system": "user",
        }

    def _invalidate_tools(self):
        """Update Gemini-side tools without mutating synaptic tools."""
        self._convert_tools()

    def _convert_tools(self) -> None:
        """Convert synaptic Tool objects + TOOL_REGISTRY to Gemini types.Tool objects with logs."""
        self.gemini_tools = []

        # if self.response_format != ResponseFormat.NONE:
        #     return
        # Use a dict to deduplicate by name
        all_tools = {}
        # Add tools from self.synaptic_tools
        for t in self.synaptic_tools or []:
            all_tools[t.name] = t
        # Add tools from TOOL_REGISTRY if not already added
        for t_name, t in TOOL_REGISTRY.items():
            if t_name not in all_tools:
                all_tools[t_name] = t
        # Convert to gemini_tools
        for t_name, t in all_tools.items():
            self.gemini_tools.append(types.Tool(function_declarations=[t.declaration]))  # type: ignore
        # Update self.synaptic_tools to include all tools
        self.synaptic_tools = list(all_tools.values())

    def to_contents(self) -> list[types.Content]:
        """Convert all memories to Gemini Content objects."""
        contents = []

        for memory in self.history.MemoryList:
            parts: list[types.Part] = [types.Part(text=memory.message)]
            parts.append(types.Part(text=f"(Created at: {memory.created})"))

            if isinstance(memory, ResponseMem):
                # Add tool calls as extra parts
                if memory.tool_calls:
                    calls_text = "Tool calls: " + str(memory.tool_calls)
                    parts.append(types.Part(text=calls_text))
                if getattr(memory, "tool_results", []):
                    results_text = "Tool results: " + str(memory.tool_results)
                    parts.append(types.Part(text=results_text))

            contents.append(
                types.Content(
                    role=(self.role_map.get(memory.role, "user")), parts=parts
                )
            )

        return contents

    def invoke(self, prompt: str, role: str = "user", **kwargs) -> ResponseMem:
        # Build config with tools if any
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            tools=self.gemini_tools,
        )

        role = self.role_map.get(role, "user")

        contents = self.to_contents()
        content = [types.Content(role=role, parts=[types.Part(text=prompt)])]
        contents = contents + content

        if self.response_format == ResponseFormat.NONE:
            config.response_mime_type = "text/plain"
        elif self.response_format == ResponseFormat.JSON:
            config.response_mime_type = "application/json"
            config.tools = None

        if self.instructions:
            instructions = [
                types.Content(
                    role=self.role_map.get("system", "user"),
                    parts=[types.Part(text=self.instructions)],
                )
            ]
            contents = instructions + contents

        # Call Gemini
        response = self.client.models.generate_content(
            model=self.model, contents=contents, config=config, **kwargs
        )

        # Extract metadata
        created = datetime.now().astimezone(timezone.utc)
        message = ""
        tool_calls: List[ToolCall] = []

        if response.candidates:
            candidate = response.candidates[0]

            # Only process content if it exists
            if candidate.content:
                if candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            message += part.text
                        if part.function_call:
                            tool_calls.append(
                                ToolCall(
                                    name=part.function_call.name,  # type: ignore
                                    args=part.function_call.args or {},
                                )
                            )

        return ResponseMem(message=message, created=created, tool_calls=tool_calls)
