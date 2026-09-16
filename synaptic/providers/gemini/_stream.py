"""GeminiStream implementation separated by runtime responsibility."""

import inspect
import asyncio
from typing import AsyncIterator, List, Optional
from google.genai import types
from ...core.base import ResponseChunk, ResponseFormat
from ...core.tool import ToolCall
from .helpers import build_audio_parts, build_image_parts


class GeminiStream:
    async def astream(
        self,
        prompt: Optional[str],
        role: str = "user",
        images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        abort=None,
        **kwargs,
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

        config = types.GenerateContentConfig(
            temperature=kwargs.pop("temperature", self.temperature),
            max_output_tokens=kwargs.pop("max_tokens", self.max_tokens),
            tools=self.gemini_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            system_instruction=self.instructions or None,
        )
        if self.response_format == ResponseFormat.NONE:
            config.response_mime_type = "text/plain"
        elif self.response_format == ResponseFormat.JSON:
            config.response_mime_type = "application/json"
            if self.response_schema is not None:
                config.response_schema = self.response_schema
            config.tools = None

        accumulated_message = ""
        tool_calls: List[ToolCall] = []
        usage_metadata = None
        finish_reason = None

        source = await self.client.aio.models.generate_content_stream(  # type: ignore[union-attr]
            model=self.model,
            contents=contents,
            config=config,
            **kwargs,  # type: ignore[arg-type]
        )
        try:
            async for response in source:
                if abort and abort.is_set():
                    raise asyncio.CancelledError()
                if getattr(response, "usage_metadata", None):
                    usage_metadata = response.usage_metadata
                candidates = getattr(response, "candidates", None)
                if not candidates:
                    continue
                candidate = candidates[0]
                finish_reason = (
                    str(candidate.finish_reason)
                    if getattr(candidate, "finish_reason", None)
                    else finish_reason
                )
                if not candidate or not candidate.content or not candidate.content.parts:
                    continue
                for part in candidate.content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        accumulated_message += text
                        yield ResponseChunk(text=text, is_final=False, function_call=None)
                    fc = getattr(part, "function_call", None)
                    if fc:
                        tfc = ToolCall(name=fc.name, args=fc.args or {}, id=getattr(fc, "id", None))  # type: ignore[union-attr]
                        tool_calls.append(tfc)
                        yield ResponseChunk(text="", is_final=False, function_call=tfc)
        finally:
            close = getattr(source, "aclose", None) or getattr(source, "close", None)
            if close:
                result = close()
                if inspect.isawaitable(result):
                    await result

        yield ResponseChunk(
            text=accumulated_message,
            is_final=True,
            function_call=None,
            input_tokens=getattr(usage_metadata, "prompt_token_count", None),
            output_tokens=getattr(usage_metadata, "candidates_token_count", None),
            finish_reason=finish_reason,
        )
