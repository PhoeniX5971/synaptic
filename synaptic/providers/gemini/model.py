from datetime import datetime, timezone
from typing import Any, List

import google.genai as genai
from dotenv import load_dotenv
from google.genai import types

from ...core.base import BaseModel, History, ResponseFormat, ResponseMem
from ...core.tool import ToolCall

load_dotenv()


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
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.tools = tools or []
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.role_map = {
            "user": "user",
            "assistant": "model",
            "system": "system",
        }

    def _convert_tools(self) -> list[types.Tool]:
        """Convert custom Tool objects to Gemini `types.Tool` objects."""
        gemini_tools = []
        for t in self.tools:
            # Each Tool may have a declaration dict
            gemini_tools.append(types.Tool(function_declarations=[t.declaration]))
        return gemini_tools

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
        tools = self._convert_tools()
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            tools=tools,
        )

        if self.response_format == ResponseFormat.NONE:
            config.response_mime_type = "text/plain"
        elif self.response_format == ResponseFormat.JSON:
            config.response_mime_type = "application/json"
            config.response_schema = self.response_schema

        role = self.role_map.get(role, "user")
        contents = self.to_contents()
        content = [types.Content(role=role, parts=[types.Part(text=prompt)])]
        contents = contents + content

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
