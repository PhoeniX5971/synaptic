"""Bounded media reads and common Chat Completions content conversion."""

import base64
import mimetypes
from pathlib import Path
from urllib.request import urlopen

MAX_MEDIA_BYTES = 16 * 1024 * 1024


def read_media(source):
    if source.startswith("data:"):
        header, data = source.split(",", 1)
        raw = base64.b64decode(data, validate=True)
        mime = header[5:].split(";")[0]
    elif source.startswith(("http://", "https://")):
        with urlopen(source, timeout=10) as response:
            raw = response.read(MAX_MEDIA_BYTES + 1)
            mime = response.headers.get_content_type()
    else:
        path = Path(source)
        with path.open("rb") as file:
            raw = file.read(MAX_MEDIA_BYTES + 1)
        mime = mimetypes.guess_type(source)[0] or "application/octet-stream"
    if len(raw) > MAX_MEDIA_BYTES:
        raise ValueError("Media exceeds 16 MiB limit")
    return raw, mime


def image_url(source):
    if source.startswith(("https://", "http://", "data:")):
        return source
    raw, mime = read_media(source)
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def chat_content(message, images=None, audio=None):
    if not images and not audio:
        return message or ""
    blocks = [{"type": "text", "text": message or ""}]
    blocks.extend(
        {"type": "image_url", "image_url": {"url": image_url(src)}} for src in images or []
    )
    for source in audio or []:
        raw, mime = read_media(source)
        blocks.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(raw).decode(),
                    "format": mime.split("/")[-1].replace("mpeg", "mp3"),
                },
            }
        )
    return blocks


def claude_content(message, images=None):
    if not images:
        return message or ""
    blocks = [{"type": "text", "text": message or ""}]
    for source in images:
        raw, mime = read_media(source)
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64.b64encode(raw).decode(),
                },
            }
        )
    return blocks
