# Core — Memory

**File:** `core/base/memory.py`

The memory module provides simple typed objects used to capture conversation
entries and model responses, and a `History` container for sliding-window
memory management.

---

## Classes

### `Memory`

**Type:** class  
A base memory object representing an entry in conversation history.

**Constructor**

```py
Memory(message: str, created, role: str)
```

**Fields**

- `message: str` — The textual content of the memory.
- `created` — Timestamp (arbitrary type, typically `datetime`).
- `role: str` — Role name (e.g. `"user"`, `"assistant"`, `"system"`).

**repr**

```py
<Memory role={role} message={message!r} created={created}>
```

---

### `ResponseMem`

**Type:** class — extends `Memory`  
Represents a model response. Stores tool-call metadata.

**Constructor**

```py
ResponseMem(message: str, created, tool_calls, tool_results: list = [], role: str = "assistant")
```

**Fields**

- `tool_calls: list[ToolCall]` — Parsed function calls returned by adapter.
- `tool_results: list` — Results populated after tool execution (list of dicts `{name, result}`).
- Inherits all `Memory` fields.

**Methods**

- `list_tool_calls()` — returns a list of names of invoked tools.
- `get_tool_call(name: str) -> ToolCall` — returns the first tool call matching `name`.

**repr**

```py
<Memory role={role} message={message!r} created={created} tool_calls={tool_calls} tool_results={tool_results}>
```

---

### `UserMem`

**Type:** class — extends `Memory`  
Thin subclass for user-originated messages. Defaults role to `"user"`.

**Constructor**

```py
UserMem(message: str, created, role: str = "user")
```

---

### `History`

**Type:** class — sliding-window container for `Memory` objects

**Constructor**

```py
History(memoryList: List[Memory] = [], size: int = 10)
```

**Fields**

- `MemoryList: list[Memory]` — ordered list of stored memories (oldest first).
- `size: int` — maximum number of entries to keep.

**Methods**

- `_size_update()` — internal; drops oldest memories when `len(MemoryList) > size`.
- `window(size: int) -> list[Memory]` — set window size (returns current list).
- `add(memory: Memory) -> None` — append a memory and enforce window size.

**Notes**

- `History` uses a simple list-backed policy (pop from front).
- The default `memoryList` parameter is mutable in the shown implementation; when adapting, prefer `None` default and create a new list in `__init__` to avoid shared state across instances.

---

## Example

```py
from synaptic.core.base.memory import Memory, ResponseMem, History

h = History(size=3)
h.add(Memory("hi", created="2025-01-01T00:00:00Z", role="user"))
h.add(ResponseMem("hey", created="2025-01-01T00:00:01Z", tool_calls=[]))
print(h.MemoryList)
```
