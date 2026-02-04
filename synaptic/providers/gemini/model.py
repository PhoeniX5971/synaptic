from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Iterator, AsyncIterator

import asyncio
import threading

import google.genai as genai
from dotenv import load_dotenv
from google.genai import types

from ...core.base import BaseModel, History, ResponseFormat, ResponseMem
from ...core.tool import TOOL_REGISTRY, Tool, ToolCall, register_callback

load_dotenv()


@dataclass
class ResponseChunk:
    """Simple chunk emitted by async stream."""
    text: str
    is_final: bool = False
    function_call: Optional[ToolCall] = None


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
        """Convert synaptic Tool objects + TOOL_REGISTRY to Gemini types.Tool objects with logs."""
        self.gemini_tools = []

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

        if self.history is None:
            return contents
        for memory in self.history.MemoryList:
            parts: list[types.Part] = [types.Part(text=memory.message + "\n")]
            # parts.append(types.Part(text=f"(Note that this entry was created at: {memory.created})//Keep this info to yourself."))

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

    # ---------- Keep the original sync invoke() exactly as you provided ----------
    def invoke(self, prompt: str, role: str = "user", **kwargs) -> ResponseMem:
        # Build config with tools if any
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
            # To avoid errors and unclear/unpredictable outputs
            # Models DO NOT ACCEPT AND WILL NOT CALL FUNCTIONS
            # if they are expected to output in formats
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

        # 3. Current prompt LAST
        contents.append(
            types.Content(
                role=self.role_map.get(role, "user"),
                parts=[types.Part(text=prompt)],
            )
        )

        # Call Gemini (non-streaming)
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

    # ---------- New: async-only streaming API ----------
    async def astream(self, prompt: str, role: str = "user", **kwargs) -> AsyncIterator[ResponseChunk]:
        """
        Asynchronously stream response chunks from Gemini.

        - Runs the provider's `generate_content_stream` synchronously in a background thread.
        - Pushes responses into an asyncio.Queue so the caller can `async for` chunks.
        - Yields ResponseChunk objects with partial text and function_call events.
        """
        role = self.role_map.get(role, "user")

        contents = self.to_contents()
        content = [types.Content(role=role, parts=[types.Part(text=prompt)])]
        contents = contents + content

        # Build config with tools if any
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
            instructions = [
                types.Content(
                    role=self.role_map.get("system", "user"),
                    parts=[types.Part(text=self.instructions)],
                )
            ]
            contents = instructions + contents

        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[Optional[Any]]" = asyncio.Queue()

        # Producer runs in a thread and pushes raw responses into the asyncio queue
        def producer():
            try:
                for response in self.client.models.generate_content_stream(
                    model=self.model, contents=contents, config=config, **kwargs
                ):
                    # Push each response into the async queue thread-safely
                    loop.call_soon_threadsafe(q.put_nowait, response)
            except Exception as e:
                # Propagate exception into the queue so consumer can handle it
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                # Sentinel to indicate end-of-stream
                loop.call_soon_threadsafe(q.put_nowait, None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        # Mirror the same parsing logic you use in invoke(), but yield incrementally
        accumulated_message = ""
        tool_calls: List[ToolCall] = []

        while True:
            item = await q.get()
            # If the producer put an Exception, raise it here
            if isinstance(item, Exception):
                raise item
            if item is None:
                # end of stream
                break

            response = item
            if not getattr(response, "candidates", None):
                continue

            candidate = response.candidates[0]

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # If there's text, yield the new piece
                    if getattr(part, "text", None):
                        piece = part.text
                        accumulated_message += piece
                        yield ResponseChunk(text=piece, is_final=False, function_call=None)

                    # If there's a function_call, emit it as a chunk
                    if getattr(part, "function_call", None):
                        fc = part.function_call
                        tfc = ToolCall(name=fc.name, args=fc.args or {})  # type: ignore
                        tool_calls.append(tfc)
                        # Represent function calls as a chunk with empty text but function_call set
                        yield ResponseChunk(text="", is_final=False, function_call=tfc)

        # Final chunk to indicate completion (you can ignore the text here if you want)
        yield ResponseChunk(text=accumulated_message, is_final=True, function_call=None)

