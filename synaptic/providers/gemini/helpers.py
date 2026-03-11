import mimetypes
from pathlib import Path
from typing import List, Optional

from google.genai import types


# -----------------------------
# MIME Detection
# -----------------------------
def guess_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


# -----------------------------
# Image → Gemini Part
# -----------------------------
def image_path_to_part(path: str) -> types.Part:
    """Convert an image file path into a Gemini inline Part."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type = guess_mime_type(path)
    data = file_path.read_bytes()

    return types.Part(
        inline_data=types.Blob(
            mime_type=mime_type,
            data=data,
        )
    )


def build_image_parts(images: Optional[List[str]]) -> List[types.Part]:
    """Convert a list of image paths into Gemini Parts."""
    if not images:
        return []

    return [image_path_to_part(img) for img in images]
