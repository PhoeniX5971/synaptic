import copy
import json
import warnings
from typing import Callable, Dict, List, Optional

from ..tool import ToolCall
from .content import ContentBlock
from .._metadata import finish_reason as normalize_finish
from . import _history_io


class Memory:
    """Single conversation entry with a role and timestamp."""

    def __init__(self, message: str, created, role: str, content=None):
        self.message = message
        self.created = created
        self.role = role
        self.content = tuple(content or ())

    def __repr__(self):
        return f"<Memory role={self.role} message={self.message!r}>"


class ResponseMem(Memory):
    """Assistant memory that may include tool calls and tool results."""

    def __init__(
        self,
        message: str,
        created,
        tool_calls: Optional[List[ToolCall]] = None,
        tool_results: Optional[List] = None,
        role: str = "assistant",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        finish_reason: Optional[str] = None,
        extensions: Optional[dict] = None,
        content=None,
    ):
        super().__init__(message, created, role, content)
        self.tool_calls: List[ToolCall] = tool_calls or []
        self.tool_results: List = tool_results or []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.extensions = dict(extensions or {})
        self.finish_reason = normalize_finish(finish_reason, self.extensions)

    def list_tool_calls(self) -> List[str]:
        return [tc.name for tc in self.tool_calls]

    def get_tool_call(self, name: str) -> ToolCall:
        matches = [tc for tc in self.tool_calls if tc.name == name]
        if not matches:
            raise IndexError(f"Tool call {name!r} not found")
        return matches[0]

    def __repr__(self):
        return (
            f"<ResponseMem role={self.role} message={self.message!r} tool_calls={self.tool_calls}>"
        )


class UserMem(Memory):
    """User or system memory entry; defaults to role `user`."""

    def __init__(self, message: str, created, role: str = "user", content=None):
        super().__init__(message, created, role, content)

    def __repr__(self):
        return f"<UserMem role={self.role} message={self.message!r}>"


from .history import History


def complete_tool_call(
    mem: ResponseMem,
    results: List,
    history: Optional[History] = None,
) -> None:
    """Set tool results and optionally ensure the response is in history."""
    mem.tool_results = _history_io.correlate(mem, results)
    if history is not None and mem not in history.entries:
        history.add(mem)
