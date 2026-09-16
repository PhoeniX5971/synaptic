"""XAIRequest implementation separated by runtime responsibility."""

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from xai_sdk.chat import (
    image as xai_image,
    file as xai_file,
    system,
    text,
    user,
)
from xai_sdk.proto import chat_pb2
from ...core.base import ResponseFormat
from .helpers import history_messages, image_to_data_url

_STRUCTURED_OUTPUT_TOOL = "__structured_output__"


class XAIRequest:
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

    def _build_user_message(
        self, prompt: Optional[str], images: Optional[List[Any]], audio: Optional[List[Any]] = None
    ) -> chat_pb2.Message:
        parts = [text(prompt or "")]
        for img in images or []:
            parts.append(xai_image(image_to_data_url(img)))
        for src in audio or []:
            file_id = self._upload_audio(src)
            parts.append(xai_file(file_id))
        return user(*parts)

    def _create_chat(
        self,
        client: Any,
        prompt: Optional[str],
        images: Optional[List[Any]],
        audio: Optional[List[Any]] = None,
        **overrides,
    ) -> Any:
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
            "max_tokens": self.max_tokens,
            "messages": messages,
            "temperature": self.temperature,
        }

        rf = self._response_format_proto()
        if rf is not None:
            kwargs["response_format"] = rf

        if self.xai_tools and self.response_format == ResponseFormat.NONE:
            kwargs["tools"] = self.xai_tools

        return client.chat.create(**{**kwargs, **overrides})
