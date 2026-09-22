import inspect
import asyncio
import threading
from typing import Any, AsyncIterator, List, Optional

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


from ...core.capabilities import Capabilities


class VertexAdapter(BaseModel):
    capabilities = Capabilities(audio=True, native_call_ids=False)

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
        max_tokens: int = 1024,
    ):
        vertex_init(project=project, location=location)

        self.model_name = model
        self.model = GenerativeModel(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history = history
        self.bound_tools = list(tools or [])
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

        self._subscription = register_callback(self._invalidate_tools, tool_registry)
        self._invalidate_tools()

    def _invalidate_tools(self):
        self._convert_tools()

    def _convert_tools(self):
        all_tools = collect_tools(self.bound_tools, self.tool_registry)
        all_declarations: List[FunctionDeclaration] = []

        for _, tool in all_tools.items():
            decl = tool.declaration

            if isinstance(decl, dict):
                decl = FunctionDeclaration(
                    name=decl.get("name"),
                    description=decl.get("description"),
                    parameters=decl.get("parameters"),
                )

            all_declarations.append(decl)

        if all_declarations:
            self.vertex_tools = [Tool(function_declarations=all_declarations)]
        else:
            self.vertex_tools = []

        self.synaptic_tools = list(all_tools.values())

    def to_contents(self) -> List[Content]:
        return history_contents(self.history, self.role_map)

    def _messages(
        self, prompt: Optional[str], role: str, audio: Optional[List[str]]
    ) -> List[Content]:
        role = self.role_map.get(role, "user")
        messages: List[Content] = self.to_contents()
        if prompt is not None or audio:
            parts = [Part.from_text(prompt or "")] + audio_parts(audio)
            messages.append(Content(role=role, parts=parts))
        if self.instructions:
            system_msg = Content(role="user", parts=[Part.from_text(self.instructions)])
            messages = [system_msg] + messages
        return messages

    def invoke(
        self,
        prompt: Optional[str],
        role: str = "user",
        audio: Optional[List[str]] = None,
        images=None,
        **kwargs,
    ) -> ResponseMem:
        self.capabilities.check("invoke", images=images, audio=audio)
        messages = self._messages(prompt, role, audio)
        response = self.model.generate_content(
            messages,
            generation_config=generation_config(
                self.response_format,
                kwargs.pop("temperature", self.temperature),
                kwargs.pop("max_tokens", self.max_tokens),
                self.response_schema,
            ),
            tools=self.vertex_tools,
            **kwargs,
        )
        return response_mem(response)

    async def astream(
        self,
        prompt: Optional[str],
        role: str = "user",
        audio: Optional[List[str]] = None,
        images=None,
        abort=None,
        **kwargs,
    ) -> AsyncIterator[ResponseChunk]:
        """Use the SDK native asynchronous streaming API."""
        self.capabilities.check("invoke", images=images, audio=audio)
        messages = self._messages(prompt, role, audio)
        config = generation_config(
            self.response_format,
            kwargs.pop("temperature", self.temperature),
            kwargs.pop("max_tokens", self.max_tokens),
            self.response_schema,
        )
        accumulated_message = ""
        tool_calls: List[ToolCall] = []
        usage_metadata = None
        finish_reason = None

        source = await self.model.generate_content_async(
            messages, generation_config=config, tools=self.vertex_tools, stream=True, **kwargs
        )
        try:
            async for response in source:
                if abort and abort.is_set():
                    raise asyncio.CancelledError()
                if getattr(response, "usage_metadata", None):
                    usage_metadata = response.usage_metadata
                if not getattr(response, "candidates", None):
                    continue

                cand = response.candidates[0]
                finish_reason = (
                    str(cand.finish_reason)
                    if getattr(cand, "finish_reason", None)
                    else finish_reason
                )

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
