import mimetypes
import urllib.request
from pathlib import Path
from typing import List, Optional

from google.genai import types


# -----------------------------
# MIME Detection
# -----------------------------
def guess_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _load_bytes(src: str) -> tuple[bytes, str]:
    """Read bytes from a local path or URL, returning (data, mime_type)."""
    p = Path(src)
    if p.exists():
        return p.read_bytes(), guess_mime_type(src)
    with urllib.request.urlopen(src) as resp:
        mime = resp.headers.get_content_type() or guess_mime_type(src)
        return resp.read(), mime


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
