"""Typed, portable content references; persistence and media loading remain optional."""

from dataclasses import dataclass, asdict
from typing import Literal


@dataclass(frozen=True)
class ContentBlock:
    kind: Literal["text", "image", "audio", "resource"]
    uri: str | None = None
    text: str | None = None
    mime_type: str | None = None

    def __post_init__(self):
        if self.kind not in ("text", "image", "audio", "resource"):
            raise ValueError("Unknown content modality")
        if self.kind == "text" and self.text is None:
            raise ValueError("Text blocks require text")
        if self.kind != "text" and not self.uri:
            raise ValueError("Media/resource blocks require a URI or local reference")

    def to_dict(self):
        return asdict(self)


def media_content(images=None, audio=None):
    return tuple(
        [ContentBlock("image", uri=src) for src in images or []]
        + [ContentBlock("audio", uri=src) for src in audio or []]
    )


def references(memory, kind):
    return [block.uri for block in getattr(memory, "content", ()) if block.kind == kind]


def message_text(memory):
    return (memory.message or "") + "".join(
        block.text or "" for block in getattr(memory, "content", ()) if block.kind == "text"
    )
