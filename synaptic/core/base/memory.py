from typing import List
from ..tool import ToolCall


class Memory:
    def __init__(self, message: str, created, role: str):
        self.message = message
        self.created = created
        self.role = role

    def __repr__(self):
        return (
            f"<Memory role={self.role} message={self.message!r} created={self.created}>"
        )


class ResponseMem(Memory):
    def __init__(
        self, message: str, created, tool_calls: List[ToolCall], tool_results=[], role="assistant"
    ):
        super().__init__(message, created, role)
        self.tool_calls = tool_calls
        self.tool_results = tool_results

    def __repr__(self):
        return f"<Memory role={self.role} message={self.message!r} created={self.created} tool_calls={self.tool_calls} tool_results={self.tool_results}>"


class UserMem(Memory):
    def __init__(self, message: str, created, role="user"):
        super().__init__(message, created, role)


class History:
    def __init__(self, memoryList: List[Memory] = [], size: int = 10):
        self.MemoryList: list[Memory] = memoryList
        self.size = size

    def _size_update(self):
        if len(self.MemoryList) > self.size:
            self.MemoryList.pop(0)

    def window(self, size: int) -> list[Memory]:
        """Modify window size."""
        self.size = size
        self._size_update()
        return self.MemoryList

    def add(self, memory: Memory) -> None:
        self.MemoryList.append(memory)
        self._size_update()
