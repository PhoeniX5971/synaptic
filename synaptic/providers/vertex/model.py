import asyncio
import mimetypes
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

from dotenv import load_dotenv
from vertexai import init as vertex_init
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import TOOL_REGISTRY, ToolCall, register_callback
from ...core.tool import Tool as ST

load_dotenv()


class VertexAdapter(BaseModel):
    def __init__(
        self,
        model: str,
        project: str,
        location: str,
        history: History | None,
        response_format: ResponseFormat,
        response_schema: Any,
        tools: Optional[List[ST]],
        api_key: str | None = None,
        temperature: float = 0.8,
        instructions: str = "",
    ):
        vertex_init(project=project, location=location)

        self.model_name = model
        self.model = GenerativeModel(model)
        self.temperature = temperature
        self.history = history
        self.synaptic_tools = list(tools or [])
        self.vertex_tools: List[Tool] = []
        self.instructions = instructions
        self.response_format = response_format
        self.response_schema = response_schema

        self.role_map = {
            "user": "user",
            "assistant": "model",
            "system": "user",
        }

        register_callback(self._invalidate_tools)
        self._invalidate_tools()

    def _invalidate_tools(self):
        self._convert_tools()

    def _convert_tools(self):
        """Convert TOOL_REGISTRY + explicit tools → Vertex Tool definitions."""
        all_tools = {}
        all_declarations: List[FunctionDeclaration] = []

        for t in self.synaptic_tools:
            all_tools[t.name] = t
        for name, t in TOOL_REGISTRY.items():
            if name not in all_tools:
                all_tools[name] = t

        for _, tool in all_tools.items():
            decl = tool.declaration

            if isinstance(decl, dict):
                decl = FunctionDeclaration(
                    name=decl.get("name"),  # type: ignore
                    description=decl.get("description"),  # type: ignore
                    parameters=decl.get("parameters"),  # type: ignore
                )

            all_declarations.append(decl)

        if all_declarations:
            self.vertex_tools = [Tool(function_declarations=all_declarations)]
        else:
            self.vertex_tools = []

        self.synaptic_tools = list(all_tools.values())

    def to_contents(self) -> List[Content]:
        contents = []

        if self.history is None:
            return contents

        for mem in self.history.MemoryList:
            if isinstance(mem, ResponseMem) and mem.tool_calls:
                # Model turn: text + function calls are implicit; Vertex only needs the text
                model_parts = [Part.from_text(mem.message)] if mem.message else [Part.from_text("")]
                contents.append(Content(role="model", parts=model_parts))

                # User turn: one FunctionResponse per call, linked by name
                tool_results = getattr(mem, "tool_results", None) or []
                if tool_results:
                    response_parts = []
                    for tc, result in zip(mem.tool_calls, tool_results):
                        resp = result.get("result", result.get("error", "")) if isinstance(result, dict) else str(result)
                        response_parts.append(
                            Part.from_function_response(
                                name=tc.name,
                                response={"result": resp},
                            )
                        )
                    contents.append(Content(role="user", parts=response_parts))
            else:
                parts = [Part.from_text(mem.message)]
                contents.append(
                    Content(role=self.role_map.get(mem.role, "user"), parts=parts)
                )

        return contents

    def _audio_parts(self, audio: Optional[List[str]]) -> List[Part]:
        parts = []
        for src in (audio or []):
            p = Path(src)
            if p.exists():
                data = p.read_bytes()
                mime, _ = mimetypes.guess_type(src)
            else:
                with urllib.request.urlopen(src) as r:
                    data = r.read()
                    mime = r.headers.get_content_type()
            if not mime or not mime.startswith("audio/"):
                mime = "audio/mpeg"
            parts.append(Part.from_data(data=data, mime_type=mime))
        return parts

    def invoke(self, prompt: str, role: str = "user", audio: Optional[List[str]] = None, **kwargs) -> ResponseMem:
        role = self.role_map.get(role, "user")

        history_contents = self.to_contents()
        user_parts = [Part.from_text(prompt)] + self._audio_parts(audio)
        user_message = Content(role=role, parts=user_parts)

        messages: List[Content] = history_contents + [user_message]

        if self.instructions:
            system_msg = Content(role="user", parts=[Part.from_text(self.instructions)])
            messages = [system_msg] + messages

        if self.response_format == ResponseFormat.NONE:
            response_mime = "text/plain"
        elif self.response_format == ResponseFormat.JSON:
            response_mime = "application/json"
        else:
            response_mime = "text/plain"

        config = GenerationConfig(
            temperature=self.temperature,
            response_mime_type=response_mime,
        )

        response = self.model.generate_content(
            messages,
            generation_config=config,
            tools=self.vertex_tools,
        )

        created = datetime.now().astimezone(timezone.utc)
        message = ""
        tool_calls: List[ToolCall] = []

        if response.candidates:
            cand = response.candidates[0]

            if cand.function_calls:
                for fc in cand.function_calls:
                    tool_calls.append(
                        ToolCall(
                            name=fc.name,
                            args=dict(fc.args) or {},
                        )
                    )

            if cand.content and cand.content.parts:
                for p in cand.content.parts:
                    if p.text:
                        message += p.text

        return ResponseMem(
            message=message,
            created=created,
            tool_calls=tool_calls,
        )

    async def astream(
        self, prompt: str, role: str = "user", audio: Optional[List[str]] = None, **kwargs
    ) -> AsyncIterator[ResponseChunk]:
        """
        Asynchronously stream response chunks from Vertex AI.

        Runs generate_content_stream synchronously in a background thread and
        pushes responses into an asyncio.Queue.
        """
        role = self.role_map.get(role, "user")

        history_contents = self.to_contents()
        user_parts = [Part.from_text(prompt)] + self._audio_parts(audio)
        user_message = Content(role=role, parts=user_parts)
        messages: List[Content] = history_contents + [user_message]

        if self.instructions:
            system_msg = Content(role="user", parts=[Part.from_text(self.instructions)])
            messages = [system_msg] + messages

        if self.response_format == ResponseFormat.NONE:
            response_mime = "text/plain"
        elif self.response_format == ResponseFormat.JSON:
            response_mime = "application/json"
        else:
            response_mime = "text/plain"

        config = GenerationConfig(
            temperature=self.temperature,
            response_mime_type=response_mime,
        )

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[Any]] = asyncio.Queue()

        def producer():
            try:
                for response in self.model.generate_content_stream(
                    messages,
                    generation_config=config,
                    tools=self.vertex_tools,
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

            cand = response.candidates[0]

            if cand.function_calls:
                for fc in cand.function_calls:
                    tfc = ToolCall(name=fc.name, args=dict(fc.args) or {})
                    tool_calls.append(tfc)
                    yield ResponseChunk(text="", is_final=False, function_call=tfc)

            if cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if part.text:
                        accumulated_message += part.text
                        yield ResponseChunk(
                            text=part.text, is_final=False, function_call=None
                        )

        yield ResponseChunk(text=accumulated_message, is_final=True, function_call=None)
