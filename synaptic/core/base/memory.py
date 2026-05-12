import copy
import json
import warnings
from typing import Callable, Dict, List, Optional

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
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        super().__init__(message, created, role)
        self.tool_calls: List[ToolCall] = tool_calls or []
        self.tool_results: List = tool_results or []
        self.input_tokens: int = input_tokens
        self.output_tokens: int = output_tokens

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
        dedup_tools: bool = False,
    ):
        self.MemoryList: List[Memory] = list(memoryList or [])
        self.size = size
        self.on_truncate = on_truncate
        self.dedup_tools = dedup_tools

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

    def effective_mems(self) -> List[Memory]:
        """Return MemoryList, optionally with stale duplicate tool results pruned.

        When dedup_tools=True, for each (tool_name, args) pair only the last
        ResponseMem that contains it is kept. Stale calls (and their results)
        are stripped; if that leaves a ResponseMem with no calls and no text it
        is dropped entirely.
        """
        if not self.dedup_tools:
            return self.MemoryList

        def _key(call: ToolCall) -> str:
            return f"{call.name}\x00{json.dumps(call.args, sort_keys=True, default=str)}"

        # Forward pass: record the last index each (name, args) key appears at.
        last: Dict[str, int] = {}
        for i, mem in enumerate(self.MemoryList):
            if isinstance(mem, ResponseMem) and mem.tool_calls:
                for call in mem.tool_calls:
                    last[_key(call)] = i

        result: List[Memory] = []
        for i, mem in enumerate(self.MemoryList):
            if not (isinstance(mem, ResponseMem) and mem.tool_calls):
                result.append(mem)
                continue

            fresh = [j for j, call in enumerate(mem.tool_calls) if last[_key(call)] == i]
            if len(fresh) == len(mem.tool_calls):
                result.append(mem)
            elif fresh:
                trimmed = copy.copy(mem)
                trimmed.tool_calls = [mem.tool_calls[j] for j in fresh]
                trimmed.tool_results = (
                    [mem.tool_results[j] for j in fresh] if mem.tool_results else []
                )
                result.append(trimmed)
            elif mem.message:
                # All calls stale but there's still text — keep as a plain reply
                trimmed = copy.copy(mem)
                trimmed.tool_calls = []
                trimmed.tool_results = []
                result.append(trimmed)
            # else: drop entirely — no text, no fresh calls

        return result


def complete_tool_call(
    mem: ResponseMem,
    results: List,
    history: Optional[History] = None,
) -> None:
    """Set tool results and optionally ensure the response is in history."""
    mem.tool_results = results
    if history is not None and mem not in history.MemoryList:
        history.add(mem)
