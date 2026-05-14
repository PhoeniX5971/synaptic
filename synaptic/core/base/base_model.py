from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import enum

from .memory import ResponseMem


class ResponseFormat(enum.StrEnum):
    JSON = "json"
    NONE = "none"


@dataclass
class ToolCallArgsDelta:
    """Partial tool-call argument fragment streamed before the call completes."""
    name: str
    delta: str      # this token's fragment
    snapshot: str   # full args accumulated so far


@dataclass
class ResponseChunk:
    """Chunk emitted by async stream."""

    text: str
    is_final: bool = False
    function_call: Optional[Any] = None
    tool_call_delta: Optional[ToolCallArgsDelta] = None
    finish_reason: Optional[str] = None   # "stop" | "tool_calls" | "max_tokens"
    input_tokens: int = 0
    output_tokens: int = 0


class BaseModel(ABC):
    @abstractmethod
    def invoke(self, prompt: Optional[str], **kwargs) -> ResponseMem:
        pass
