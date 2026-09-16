import mimetypes
import urllib.request
from pathlib import Path
from typing import List, Optional

from google.genai import types

from ...core.base import ResponseMem
from ...core.base.content import references, message_text
from .._media import read_media


# -----------------------------
# MIME Detection
# -----------------------------
def guess_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _load_bytes(src: str) -> tuple[bytes, str]:
    """Read bytes from a local path or URL, returning (data, mime_type)."""
    return read_media(src)


# -----------------------------
# Image → Gemini Part
# -----------------------------
def image_path_to_part(path: str) -> types.Part:
    """Convert an image file path or URL into a Gemini inline Part."""
    data, mime_type = _load_bytes(path)
    return types.Part(inline_data=types.Blob(mime_type=mime_type, data=data))


def build_image_parts(images: Optional[List[str]]) -> List[types.Part]:
    """Convert a list of image paths/URLs into Gemini Parts."""
    if not images:
        return []
    return [image_path_to_part(img) for img in images]


# -----------------------------
# Audio → Gemini Part
# -----------------------------
def audio_to_part(src: str) -> types.Part:
    """Convert an audio file path or URL into a Gemini inline Part."""
    data, mime_type = _load_bytes(src)
    if not mime_type.startswith("audio/"):
        mime_type = "audio/mpeg"
    return types.Part(inline_data=types.Blob(mime_type=mime_type, data=data))


def build_audio_parts(audio: Optional[List[str]]) -> List[types.Part]:
    """Convert a list of audio paths/URLs into Gemini Parts."""
    if not audio:
        return []
    return [audio_to_part(src) for src in audio]


def history_contents(history, role_map) -> list[types.Content]:
    contents = []
    if history is None:
        return contents
    for memory in history.effective_mems():
        if isinstance(memory, ResponseMem) and memory.tool_calls:
            parts: list[types.Part] = []
            if message_text(memory):
                parts.append(types.Part(text=message_text(memory)))
            for call in memory.tool_calls:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(name=call.name, args=call.args, id=call.id)
                    )
                )
            contents.append(types.Content(role="model", parts=parts))
            results = getattr(memory, "tool_results", None) or []
            response_parts: list[types.Part] = []
            for call, result in zip(memory.tool_calls, results):
                resp = (
                    result.get("result", result.get("error", ""))
                    if isinstance(result, dict)
                    else str(result)
                )
                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=call.name, response={"result": resp}, id=call.id
                        )
                    )
                )
            if response_parts:
                contents.append(types.Content(role="user", parts=response_parts))
        else:
            contents.append(
                types.Content(
                    role=role_map.get(memory.role, "user"),
                    parts=[types.Part(text=message_text(memory))]
                    + build_image_parts(references(memory, "image"))
                    + build_audio_parts(references(memory, "audio")),
                )
            )
    return contents
