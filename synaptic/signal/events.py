from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

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
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class BlockStarted:
    type: str
    attrs: Dict[str, str] = field(default_factory=dict)


@dataclass
class BlockDelta:
    type: str
    delta: str
    snapshot: str
    attrs: Dict[str, str] = field(default_factory=dict)


@dataclass
class BlockDone:
    type: str
    content: str
    attrs: Dict[str, str] = field(default_factory=dict)


# ToolCallArgsDelta is re-exported from core — it doubles as a signal event.
__all__ = [
    "TextDelta",
    "ToolCallStarted",
    "ToolCallArgsDelta",
    "ToolCallDone",
    "ToolCallResult",
    "TurnComplete",
    "BlockStarted",
    "BlockDelta",
    "BlockDone",
    "SignalEvent",
]

SignalEvent = Union[
    TextDelta,
    ToolCallStarted,
    ToolCallArgsDelta,
    ToolCallDone,
    ToolCallResult,
    TurnComplete,
    BlockStarted,
    BlockDelta,
    BlockDone,
]
