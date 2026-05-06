import asyncio
import threading
from typing import Any, AsyncIterator, List, Optional

from dotenv import load_dotenv
from vertexai import init as vertex_init
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
)

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import ToolCall, ToolRegistry, collect_tools, register_callback
from ...core.tool import Tool as ST
from .helpers import audio_parts, generation_config, history_contents, response_mem

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
        tool_registry: Optional[ToolRegistry] = None,
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
        self.tool_registry = tool_registry

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
        all_tools = collect_tools(self.synaptic_tools, self.tool_registry)
        all_declarations: List[FunctionDeclaration] = []

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
        return history_contents(self.history, self.role_map)

    def _messages(self, prompt: Optional[str], role: str, audio: Optional[List[str]]) -> List[Content]:
        role = self.role_map.get(role, "user")
        messages: List[Content] = self.to_contents()
        if prompt is not None or audio:
            parts = [Part.from_text(prompt or "")] + audio_parts(audio)
            messages.append(Content(role=role, parts=parts))
        if self.instructions:
            system_msg = Content(role="user", parts=[Part.from_text(self.instructions)])
            messages = [system_msg] + messages
        return messages

    def invoke(self, prompt: Optional[str], role: str = "user", audio: Optional[List[str]] = None, **kwargs) -> ResponseMem:
        messages = self._messages(prompt, role, audio)
        response = self.model.generate_content(
            messages,
            generation_config=generation_config(self.response_format, self.temperature),
            tools=self.vertex_tools,
        )
        return response_mem(response)

    async def astream(
        self, prompt: Optional[str], role: str = "user", audio: Optional[List[str]] = None,
        abort=None, **kwargs
    ) -> AsyncIterator[ResponseChunk]:
        """
        Asynchronously stream response chunks from Vertex AI.

        Runs generate_content_stream synchronously in a background thread and
        pushes responses into an asyncio.Queue.
        """
        messages = self._messages(prompt, role, audio)
        config = generation_config(self.response_format, self.temperature)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[Any]] = asyncio.Queue()

        def producer():
            try:
                for response in self.model.generate_content_stream(
                    messages,
                    generation_config=config,
                    tools=self.vertex_tools,
                ):
                    if abort and abort.is_set():
                        break
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
            if abort and abort.is_set():
                return
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
