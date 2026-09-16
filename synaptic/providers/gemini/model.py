from ._stream import GeminiStream
from datetime import datetime, timezone
from typing import Any, List, Optional

import google.genai as genai
from google.genai import types

from ...core.base import BaseModel, History, ResponseFormat, ResponseMem
from ...core.tool import Tool, ToolCall, ToolRegistry, collect_tools, register_callback
from .helpers import build_audio_parts, build_image_parts, history_contents


from ...core.capabilities import Capabilities


class GeminiAdapter(GeminiStream, BaseModel):
    capabilities = Capabilities(images=True, audio=True)

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
        tool_registry: Optional[ToolRegistry] = None,
        max_tokens: int = 1024,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.bound_tools = list(tools or [])
        self.synaptic_tools = list(tools or [])
        self.gemini_tools: List[types.Tool] = []
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.tool_registry = tool_registry
        self._subscription = register_callback(self._invalidate_tools, tool_registry)
        self._invalidate_tools()
        self.instructions = instructions
        self.role_map = {
            "user": "user",
            "assistant": "model",
            "system": "user",
        }

    def _invalidate_tools(self):
        self._convert_tools()

    def _convert_tools(self) -> None:
        self.gemini_tools = []
        all_tools = collect_tools(self.bound_tools, self.tool_registry)
        for t in all_tools.values():
            self.gemini_tools.append(types.Tool(function_declarations=[t.declaration]))
        self.synaptic_tools = list(all_tools.values())

    def to_contents(self) -> list[types.Content]:
        return history_contents(self.history, self.role_map)

    def invoke(
        self,
        prompt: Optional[str],
        role: str = "user",
        images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        **kwargs,
    ) -> ResponseMem:
        config = types.GenerateContentConfig(
            temperature=kwargs.pop("temperature", self.temperature),
            max_output_tokens=kwargs.pop("max_tokens", self.max_tokens),
            tools=list(self.gemini_tools),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            system_instruction=self.instructions or None,
        )
        role = self.role_map.get(role, "user")
        contents = []

        if self.response_format == ResponseFormat.NONE:
            config.response_mime_type = "text/plain"
        elif self.response_format == ResponseFormat.JSON:
            config.response_mime_type = "application/json"
            if self.response_schema is not None:
                config.response_schema = self.response_schema
            config.tools = None

        # Memory + history
        contents.extend(self.to_contents())

        # 3. Current prompt LAST (text + optional images)
        prompt_parts: list[types.Part] = []

        if prompt and prompt.strip():
            prompt_parts.append(types.Part(text=prompt))

        prompt_parts.extend(build_image_parts(images))
        prompt_parts.extend(build_audio_parts(audio))

        if prompt_parts:
            contents.append(types.Content(role=role, parts=prompt_parts))

        sdk_contents: list[types.ContentUnionDict] = list(contents)
        response = self.client.models.generate_content(
            model=self.model, contents=sdk_contents, config=config, **kwargs
        )

        created = datetime.now().astimezone(timezone.utc)
        message = ""
        tool_calls: List[ToolCall] = []

        if response.candidates:
            candidate = response.candidates[0]

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        message += part.text
                    if part.function_call:
                        call = part.function_call
                        if not call.name:
                            raise ValueError("Gemini function call is missing its name")
                        tool_calls.append(
                            ToolCall(name=call.name, args=call.args or {}, id=call.id)
                        )

        um = getattr(response, "usage_metadata", None)
        return ResponseMem(
            message=message,
            created=created,
            tool_calls=tool_calls,
            input_tokens=getattr(um, "prompt_token_count", None),
            output_tokens=getattr(um, "candidates_token_count", None),
            finish_reason=str(response.candidates[0].finish_reason)
            if response.candidates
            else None,
        )
