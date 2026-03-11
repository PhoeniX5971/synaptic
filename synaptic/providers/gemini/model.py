import asyncio
import threading
from datetime import datetime, timezone
from typing import Any, AsyncIterator, List, Optional

import google.genai as genai
from dotenv import load_dotenv
from google.genai import types

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import TOOL_REGISTRY, Tool, ToolCall, register_callback
from .helpers import build_image_parts

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
        stream: bool = False,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.synaptic_tools = list(tools or [])
        self.gemini_tools: List[types.Tool] = []
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        register_callback(self._invalidate_tools)
        self._invalidate_tools()
        self.instructions = instructions
        # keep this flag for backwards compatibility / config, but streaming will be done via astream()
        self.stream = stream
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
        all_tools = {}
        for t in self.synaptic_tools or []:
            all_tools[t.name] = t
        for t_name, t in TOOL_REGISTRY.items():
            if t_name not in all_tools:
                all_tools[t_name] = t

        for t_name, t in all_tools.items():
            self.gemini_tools.append(types.Tool(function_declarations=[t.declaration]))  # type: ignore

        self.synaptic_tools = list(all_tools.values())

    def to_contents(self) -> list[types.Content]:
        """Convert all memories to Gemini Content objects."""
        contents = []

        if self.history is None:
            return contents
        for memory in self.history.MemoryList:
            parts: list[types.Part] = [types.Part(text=memory.message + "\n")]

            if isinstance(memory, ResponseMem):
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

    def invoke(
        self,
        prompt: str,
        role: str = "user",
        images: Optional[List[str]] = None,
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

        if prompt.strip():
            prompt_parts.append(types.Part(text=prompt))

        prompt_parts.extend(build_image_parts(images))

        contents.append(
            types.Content(
                role=self.role_map.get(role, "user"),
                parts=prompt_parts,
            )
        )

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
        self, prompt: str, role: str = "user", **kwargs
    ) -> AsyncIterator[ResponseChunk]:
        """
        Asynchronously stream response chunks from Gemini.

        Runs generate_content_stream synchronously in a background thread,
        pushes responses into an asyncio.Queue so the caller can async-for chunks.
        Yields ResponseChunk objects with partial text and function_call events.
        """
        role = self.role_map.get(role, "user")

        contents = self.to_contents()
        contents.append(types.Content(role=role, parts=[types.Part(text=prompt)]))

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            tools=self.gemini_tools,
        )

        if self.response_format == ResponseFormat.NONE:
            config.response_mime_type = "text/plain"
        elif self.response_format == ResponseFormat.JSON:
            config.response_mime_type = "application/json"
            if self.response_schema is not None:
                config.response_schema = self.response_schema
            config.tools = None

        if self.instructions:
            instructions_content = [
                types.Content(
                    role=self.role_map.get("system", "user"),
                    parts=[types.Part(text=self.instructions)],
                )
            ]
            contents = instructions_content + contents

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[Any]] = asyncio.Queue()

        def producer():
            try:
                for response in self.client.models.generate_content_stream(
                    model=self.model, contents=contents, config=config, **kwargs  # type: ignore
                ):
                    loop.call_soon_threadsafe(q.put_nowait, response)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=producer, daemon=True).start()

        accumulated_message = ""
        tool_calls: List[ToolCall] = []

        while True:
            item = await q.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                break

            response = item
            if not getattr(response, "candidates", None):
                continue

            candidate = response.candidates[0]

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, "text", None):
                        piece = part.text
                        accumulated_message += piece
                        yield ResponseChunk(
                            text=piece, is_final=False, function_call=None
                        )

                    if getattr(part, "function_call", None):
                        fc = part.function_call
                        tfc = ToolCall(name=fc.name, args=fc.args or {})  # type: ignore
                        tool_calls.append(tfc)
                        yield ResponseChunk(text="", is_final=False, function_call=tfc)

        yield ResponseChunk(text=accumulated_message, is_final=True, function_call=None)
