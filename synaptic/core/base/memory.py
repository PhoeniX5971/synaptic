import warnings
from typing import Callable, List, Optional

from ..tool import ToolCall


class Memory:
    """Single conversation entry with a role and timestamp."""

    def __init__(self, message: str, created, role: str):
        self.message = message
        self.created = created
        self.role = role

    def __repr__(self):
        return f"<Memory role={self.role} message={self.message!r}>"


class ResponseMem(Memory):
    """Assistant memory that may include tool calls and tool results."""

    def __init__(
        self,
        message: str,
        created,
        tool_calls: List[ToolCall] = None,
        tool_results: List = None,
        role: str = "assistant",
    ):
        super().__init__(message, created, role)
        self.tool_calls: List[ToolCall] = tool_calls or []
        self.tool_results: List = tool_results or []

    def list_tool_calls(self) -> List[str]:
        return [tc.name for tc in self.tool_calls]

    def get_tool_call(self, name: str) -> ToolCall:
        matches = [tc for tc in self.tool_calls if tc.name == name]
        if not matches:
            raise IndexError(f"Tool call {name!r} not found")
        return matches[0]

    def __repr__(self):
        return f"<ResponseMem role={self.role} message={self.message!r} tool_calls={self.tool_calls}>"


class UserMem(Memory):
    """User or system memory entry; defaults to role `user`."""

    def __init__(self, message: str, created, role: str = "user"):
        super().__init__(message, created, role)

    def __repr__(self):
        return f"<UserMem role={self.role} message={self.message!r}>"


class History:
    """Sliding window of conversation memories.

    When the window drops entries, `on_truncate` receives the removed memories.
    Without a callback, truncation emits a warning so lost context is visible.
    """

    def __init__(
        self,
        memoryList: Optional[List[Memory]] = None,
        size: int = 10,
        on_truncate: Optional[Callable[[List[Memory]], None]] = None,
    ):
        self.MemoryList: List[Memory] = list(memoryList or [])
        self.size = size
        self.on_truncate = on_truncate

    def _size_update(self) -> None:
        excess = len(self.MemoryList) - self.size
        if excess > 0:
            dropped = self.MemoryList[:excess]
            self.MemoryList = self.MemoryList[excess:]
            if self.on_truncate:
                self.on_truncate(dropped)
            else:
                warnings.warn(
                    f"History truncated: {len(dropped)} "
                    f"entr{'y' if len(dropped) == 1 else 'ies'} dropped. "
                    "Pass on_truncate= to handle this explicitly.",
                    stacklevel=3,
                )

    def window(self, size: int) -> List[Memory]:
        self.size = size
        self._size_update()
        return self.MemoryList

    def add(self, memory: Memory) -> None:
        self.MemoryList.append(memory)
        self._size_update()


def complete_tool_call(
    mem: ResponseMem,
    results: List,
    history: Optional[History] = None,
) -> None:
    """Set tool results and optionally ensure the response is in history."""
    mem.tool_results = results
    if history is not None and mem not in history.MemoryList:
        history.add(mem)
