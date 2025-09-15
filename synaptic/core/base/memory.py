class Memory:
    def __init__(self, message: str, created, role: str):
        self.message = message
        self.created = created
        self.role = role


class ResponseMem(Memory):
    def __init__(
        self, message: str, created, tool_calls, tool_results=None, role="assistant"
    ):
        super().__init__(message, created, role)
        self.tool_calls = tool_calls
        self.tool_results = tool_results


class UserMem(Memory):
    def __init__(self, message: str, created, role="user"):
        super().__init__(message, created, role)


class History:
    MemoryList: list[Memory] = []

    def window(self, size: int) -> list[Memory]:
        """Get the most recent `size` memories."""
        return self.MemoryList[-size:] if size > 0 else self.MemoryList
