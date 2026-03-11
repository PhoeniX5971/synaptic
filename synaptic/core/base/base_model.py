from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .memory import ResponseMem
import enum


class ResponseFormat(enum.StrEnum):
    JSON = "json"
    NONE = "none"


@dataclass
class ResponseChunk:
    """Simple chunk emitted by async stream."""

    text: str
    is_final: bool = False
    function_call: Optional[Any] = None


class BaseModel(ABC):
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> ResponseMem:
        pass
