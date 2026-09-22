"""Count and token-bounded history windows of complete memory entries."""

import copy
import json
import warnings
from typing import Any, Callable, Dict, List, Optional
from ..tool import ToolCall
from .._awaitables import require_sync
from .memory import Memory, ResponseMem
from . import _history_io


class History:
    """Sliding window of conversation memories.

    When the window drops entries, `on_truncate` receives the removed memories.
    Without a callback, truncation emits a warning so lost context is visible.
    """

    def __init__(
        self,
        memory_list: Optional[List[Memory]] = None,
        size: int = 10,
        on_truncate: Optional[Callable[[List[Memory]], None]] = None,
        dedup_tools: bool = False,
        token_budget: Optional[int] = None,
        token_counter=None,
    ):
        if size < 1 or (
            token_budget is not None and (token_budget < 1 or token_counter is None)
        ):
            raise ValueError("Positive history size/budget and explicit token_counter required")
        self.token_budget, self.token_counter = token_budget, token_counter
        self.entries: List[Memory] = list(memory_list or [])
        self.recovery: list[ResponseMem] = []
        self.failed_attempts: list[ResponseMem] = []
        for memory in self.entries:
            self._retain(memory)
        self.size = size
        self.on_truncate = on_truncate
        self.dedup_tools = dedup_tools

    def _retain(self, memory: Memory) -> None:
        memory._histories.add(self)
        if (
            isinstance(memory, ResponseMem)
            and (memory.tool_calls or memory.extensions.get("incomplete"))
            and memory not in self.recovery
        ):
            self.recovery.append(memory)

    def enforce_budget(self) -> None:
        """Trim provider context as whole entries; recovery evidence is independent."""
        self._size_update()

    def _size_update(self) -> None:
        excess = max(0, len(self.entries) - self.size)
        if self.token_budget is not None:
            while (
                excess < len(self.entries)
                and sum(self.token_counter(m) for m in self.entries[excess:])
                > self.token_budget
            ):
                excess += 1
        # Calls and results reside in one ResponseMem: never split that exchange.
        if excess > 0:
            dropped = self.entries[:excess]
            self.entries = self.entries[excess:]
            if self.on_truncate:
                require_sync(
                    self.on_truncate(dropped),
                    "History truncation callbacks must be synchronous",
                )
            else:
                warnings.warn(
                    f"History truncated: {len(dropped)} "
                    f"entr{'y' if len(dropped) == 1 else 'ies'} dropped. "
                    "Pass on_truncate= to handle this explicitly.",
                    stacklevel=3,
                )

    def window(self, size: int) -> List[Memory]:
        if size < 1:
            raise ValueError("History size must be positive")
        self.size = size
        self._size_update()
        return self.entries

    def add(self, memory: Memory) -> None:
        self._retain(memory)
        self.entries.append(memory)
        self._size_update()

    def to_dict(self) -> dict[str, Any]:
        return _history_io.serialize(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "History":
        return _history_io.deserialize(value)

    def effective_mems(self) -> List[Memory]:
        """Return entries, optionally with stale duplicate tool results pruned.

        When dedup_tools=True, for each (tool_name, args) pair only the last
        ResponseMem that contains it is kept. Stale calls (and their results)
        are stripped; if that leaves a ResponseMem with no calls and no text it
        is dropped entirely.
        """
        self.enforce_budget()
        if not self.dedup_tools:
            return list(self.entries)

        def _key(call: ToolCall) -> str:
            return f"{call.name}\x00{json.dumps(call.args, sort_keys=True, default=str)}"

        # Forward pass: record the last index each (name, args) key appears at.
        last: Dict[str, int] = {}
        for i, mem in enumerate(self.entries):
            if isinstance(mem, ResponseMem) and mem.tool_calls:
                for call in mem.tool_calls:
                    last[_key(call)] = i

        result: List[Memory] = []
        for i, mem in enumerate(self.entries):
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
