from dataclasses import dataclass
from typing import Any, List, Optional, Union

from ..core.base.base_model import ToolCallArgsDelta
from ..core.tool import ToolCall


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallStarted:
    name: str


@dataclass
class ToolCallDone:
    call: ToolCall


@dataclass
class ToolCallResult:
    call: ToolCall
    result: Any
    error: Optional[str] = None


@dataclass
class TurnComplete:
    message: str
    tool_calls: List[ToolCall]


# ToolCallArgsDelta is re-exported from core — it doubles as a signal event.
__all__ = [
    "TextDelta",
    "ToolCallStarted",
    "ToolCallArgsDelta",
    "ToolCallDone",
    "ToolCallResult",
    "TurnComplete",
    "SignalEvent",
]

SignalEvent = Union[
    TextDelta,
    ToolCallStarted,
    ToolCallArgsDelta,
    ToolCallDone,
    ToolCallResult,
    TurnComplete,
]
