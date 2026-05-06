import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from dotenv import load_dotenv

from xai_sdk.aio.client import Client as AsyncClient
from xai_sdk.sync.client import Client as SyncClient
from xai_sdk.chat import (
    image as xai_image,
    file as xai_file,
    system,
    text,
    tool as xai_tool,
    user,
)
from xai_sdk.proto import chat_pb2

from ...core.base import BaseModel, History, ResponseChunk, ResponseFormat, ResponseMem
from ...core.tool import Tool, ToolRegistry, collect_tools, register_callback
from .helpers import history_messages, image_to_data_url, parse_calls

load_dotenv()

_STRUCTURED_OUTPUT_TOOL = "__structured_output__"


def _make_xai_tool(t: Tool) -> chat_pb2.Tool:
    decl = t.declaration
    return xai_tool(
        name=decl["name"],
        description=decl.get("description", ""),
        parameters=decl.get("parameters", {"type": "object", "properties": {}}),
    )


class XAIAdapter(BaseModel):
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
        self.model = model
        self.api_key = api_key or None
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.instructions = instructions
        self.tool_registry = tool_registry

        self.synaptic_tools = list(tools or [])
        self.xai_tools: List[chat_pb2.Tool] = []

        self._sync_client = SyncClient(api_key=self.api_key)
        self._async_client = AsyncClient(api_key=self.api_key)

        register_callback(self._invalidate_tools)
        self._invalidate_tools()

    def _invalidate_tools(self) -> None:
        self._convert_tools()

    def _convert_tools(self) -> None:
        all_tools = collect_tools(self.synaptic_tools, self.tool_registry)
        self.xai_tools = [_make_xai_tool(t) for t in all_tools.values()]
        self.synaptic_tools = list(all_tools.values())

    def _schema(self) -> Optional[Dict]:
        if self.response_schema is None:
            return None
        if hasattr(self.response_schema, "model_json_schema"):
            return self.response_schema.model_json_schema()
        return self.response_schema

    def _response_format_proto(self) -> Optional[chat_pb2.ResponseFormat]:
        if self.response_format != ResponseFormat.JSON:
            return None
        schema = self._schema()
        if schema:
            return chat_pb2.ResponseFormat(
                format_type=chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA,
                schema=json.dumps(schema),
            )
        return chat_pb2.ResponseFormat(format_type=chat_pb2.FormatType.FORMAT_TYPE_JSON_OBJECT)

    def _build_history_messages(self) -> List[chat_pb2.Message]:
        return history_messages(self.history)

    def _upload_audio(self, src: Any) -> str:
        """Upload an audio file and return its file_id."""
        src_str = str(src)
        p = Path(src_str)
        if p.exists():
            return self._sync_client.files.upload(src_str).id
        # URL — download then upload
        with urllib.request.urlopen(src_str) as r:
            data = r.read()
        from io import BytesIO
        filename = Path(src_str.split("?")[0]).name or "audio.mp3"
        return self._sync_client.files.upload(BytesIO(data), filename=filename).id

    def _build_user_message(self, prompt: Optional[str], images: Optional[List[Any]], audio: Optional[List[Any]] = None) -> chat_pb2.Message:
        parts = [text(prompt or "")]
        for img in (images or []):
            parts.append(xai_image(image_to_data_url(img)))
        for src in (audio or []):
            file_id = self._upload_audio(src)
            parts.append(xai_file(file_id))
        return user(*parts)

    def _create_chat(self, client: Any, prompt: Optional[str], images: Optional[List[Any]], audio: Optional[List[Any]] = None) -> Any:
        messages: List[chat_pb2.Message] = []

        sys_text = self.instructions or ""
        if self.response_format == ResponseFormat.JSON and not self._schema():
            sys_text += "\nYou must respond ONLY with valid JSON."
        if sys_text:
            messages.append(system(text(sys_text)))

        messages.extend(self._build_history_messages())
        if prompt is not None or images or audio:
            messages.append(self._build_user_message(prompt, images, audio))

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        rf = self._response_format_proto()
        if rf is not None:
            kwargs["response_format"] = rf

        if self.xai_tools and self.response_format == ResponseFormat.NONE:
            kwargs["tools"] = self.xai_tools

        return client.chat.create(**kwargs)

    def _parse_response(self, response: Any) -> ResponseMem:
        created = datetime.now().astimezone(timezone.utc)
        message = response.content or ""
        tool_calls = parse_calls(response)
        if self.response_format == ResponseFormat.JSON and not tool_calls:
            try:
                message = json.dumps(json.loads(message))
            except Exception:
                pass

        return ResponseMem(message=message, created=created, tool_calls=tool_calls)

    def invoke(
        self, prompt: Optional[str], role: str = "user", images: Optional[List[Any]] = None,
        audio: Optional[List[Any]] = None, **kwargs
    ) -> ResponseMem:
        chat = self._create_chat(self._sync_client, prompt, images, audio)
        response = chat.sample()
        return self._parse_response(response)

    async def astream(
        self, prompt: Optional[str], role: str = "user", images: Optional[List[Any]] = None,
        audio: Optional[List[Any]] = None, abort=None, **kwargs
    ) -> AsyncIterator[ResponseChunk]:
        chat = self._create_chat(self._async_client, prompt, images, audio)

        accumulated = ""
        async for response, chunk in chat.stream():
            if abort and abort.is_set():
                return
            for choice in chunk.choices:
                piece = choice.content
                if piece:
                    accumulated += piece
                    yield ResponseChunk(text=piece, is_final=False, function_call=None)

        for tfc in parse_calls(response):
            yield ResponseChunk(text="", is_final=False, function_call=tfc)

        yield ResponseChunk(text=accumulated, is_final=True, function_call=None)
