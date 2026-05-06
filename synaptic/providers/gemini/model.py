from datetime import datetime, timezone
from typing import Any, AsyncIterator, List, Optional

import google.genai as genai
from dotenv import load_dotenv
from google.genai import types

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import Tool, ToolCall, ToolRegistry, collect_tools, register_callback
from .helpers import build_audio_parts, build_image_parts, history_contents

load_dotenv()


class GeminiAdapter(BaseModel):
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
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.synaptic_tools = list(tools or [])
        self.gemini_tools: List[types.Tool] = []
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.tool_registry = tool_registry
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
        """Convert synaptic Tool objects + TOOL_REGISTRY to Gemini types.Tool objects."""
        self.gemini_tools = []

        # Use a dict to deduplicate by name
        all_tools = collect_tools(self.synaptic_tools, self.tool_registry)

        for t_name, t in all_tools.items():
            self.gemini_tools.append(types.Tool(function_declarations=[t.declaration]))  # type: ignore

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
            temperature=self.temperature,
            tools=self.gemini_tools,
        )

        role = self.role_map.get(role, "user")

        contents = []

        if self.response_format == ResponseFormat.NONE:
            config.response_mime_type = "text/plain"
        elif self.response_format == ResponseFormat.JSON:
            config.response_mime_type = "application/json"
            if self.response_schema is not None:
                config.response_schema = self.response_schema
            # Models will not call functions when expected to output structured formats
            config.tools = None

        # 1. System instructions FIRST
        if self.instructions:
            contents.append(
                types.Content(
                    role=self.role_map.get("system", "user"),
                    parts=[types.Part(text=self.instructions)],
                )
            )

        # 2. Memory + history
        contents.extend(self.to_contents())

        # 3. Current prompt LAST (text + optional images)
        prompt_parts: list[types.Part] = []

        if prompt and prompt.strip():
            prompt_parts.append(types.Part(text=prompt))

        prompt_parts.extend(build_image_parts(images))
        prompt_parts.extend(build_audio_parts(audio))

        if prompt_parts:
            contents.append(types.Content(role=self.role_map.get(role, "user"), parts=prompt_parts))

        response = self.client.models.generate_content(
            model=self.model, contents=contents, config=config, **kwargs
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
                        tool_calls.append(
                            ToolCall(
                                name=part.function_call.name,  # type: ignore
                                args=part.function_call.args or {},
                            )
                        )

        return ResponseMem(message=message, created=created, tool_calls=tool_calls)

    async def astream(
        self, prompt: Optional[str], role: str = "user", images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None, abort=None, **kwargs
    ) -> AsyncIterator[ResponseChunk]:
        role = self.role_map.get(role, "user")

        prompt_parts: list[types.Part] = []
        if prompt and prompt.strip():
            prompt_parts.append(types.Part(text=prompt))
        prompt_parts.extend(build_image_parts(images))
        prompt_parts.extend(build_audio_parts(audio))

        contents = self.to_contents()
        if prompt_parts:
            contents.append(types.Content(role=role, parts=prompt_parts))

        config = types.GenerateContentConfig(temperature=self.temperature, tools=self.gemini_tools)
        if self.response_format == ResponseFormat.NONE:
            config.response_mime_type = "text/plain"
        elif self.response_format == ResponseFormat.JSON:
            config.response_mime_type = "application/json"
            if self.response_schema is not None:
                config.response_schema = self.response_schema
            config.tools = None

        if self.instructions:
            contents = [types.Content(
                role=self.role_map.get("system", "user"),
                parts=[types.Part(text=self.instructions)],
            )] + contents

        accumulated_message = ""
        tool_calls: List[ToolCall] = []

        async for response in self.client.aio.models.generate_content_stream(  # type: ignore[union-attr]
            model=self.model, contents=contents, config=config, **kwargs  # type: ignore[arg-type]
        ):
            if abort and abort.is_set():
                return
            if not getattr(response, "candidates", None):
                continue
            candidate = response.candidates[0]
            if not candidate or not candidate.content or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                text = getattr(part, "text", None)
                if text:
                    accumulated_message += text
                    yield ResponseChunk(text=text, is_final=False, function_call=None)
                fc = getattr(part, "function_call", None)
                if fc:
                    tfc = ToolCall(name=fc.name, args=fc.args or {})  # type: ignore[union-attr]
                    tool_calls.append(tfc)
                    yield ResponseChunk(text="", is_final=False, function_call=tfc)

        yield ResponseChunk(text=accumulated_message, is_final=True, function_call=None)
