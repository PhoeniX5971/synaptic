# Synaptic Production Hardening Plan

## Context

Synaptic is entering production use across multiple companies. A comprehensive audit identified 13 gaps ranging from showstopper bugs (history desync, no agentic loop) to medium/low issues (global tool registry, silent history truncation, missing session/permission/event abstractions). This plan addresses all 13, restructures oversized files to comply with the 200-line limit, and introduces a new `Agent` layer above `Model`.

---

## Decisions

- **Scope**: All 13 issues
- **Loop location**: New `Agent` class in `synaptic/agent/`
- **Async refactor**: Now — native async for adapters with confirmed async SDKs; threading kept only for Together/Vertex with cancellation added
- **model.py split**: By concern — `model.py` (init/props), `_runner.py` (invoke/stream/tools), `_factory.py` (adapter selection)

---

## File Map

### New files

| File                         | Purpose                                              | Est. lines |
| ---------------------------- | ---------------------------------------------------- | ---------- |
| `synaptic/agent/__init__.py` | exports                                              | ~10        |
| `synaptic/agent/agent.py`    | `Agent` class — multi-turn loop, permissions, events | ~180       |
| `synaptic/agent/session.py`  | `Session` dataclass                                  | ~50        |
| `synaptic/agent/events.py`   | `EventBus` — lightweight subscription dict           | ~40        |

### Split: `synaptic/core/model.py` → 3 files

| File          | Moves in                                                         | Est. lines |
| ------------- | ---------------------------------------------------------------- | ---------- |
| `model.py`    | `__init__`, properties (`history`, `instructions`), `bind_tools` | ~100       |
| `_factory.py` | `_initiate_model` provider switch                                | ~110       |
| `_runner.py`  | `invoke`, `ainvoke`, `astream`, `_run_tools`, `_arun_tools`      | ~150       |

`Model` stays the single public class; `_factory` and `_runner` export module-level functions that `Model` delegates to.

### Modified files

| File                      | Changes                                                                                                                |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `core/base/base_model.py` | Add `finish_reason`, `input_tokens`, `output_tokens` to `ResponseChunk`                                                |
| `core/base/memory.py`     | Add `on_truncate` callback + `warnings.warn` to `History._size_update`; add `complete_tool_call(mem, results)` utility |
| `core/tool.py`            | Add `ToolRegistry` class; keep global `TOOL_REGISTRY` as default instance for backward compat                          |
| `synaptic/__init__.py`    | Export `Agent`, `Session`, `ToolRegistry`                                                                              |
| All adapters              | Native async (see below) + `abort: asyncio.Event` param on `astream`                                                   |

---

## Issue-by-Issue Implementation

### 1. History desync — `core/model.py`

Add `history` as a property with a setter that syncs to `self.llm.history`, identical pattern to `instructions`:

```python
@history.setter
def history(self, v: History) -> None:
    self._history = v
    if hasattr(self, "llm"):
        self.llm.history = v
```

### 2. Multi-turn tool loop — `agent/agent.py`

`Agent` wraps a `Model` and owns the loop:

```python
class Agent:
    def __init__(self, model, max_turns=10, permission=None, events=None): ...
    def run(self, prompt, **kwargs) -> ResponseMem: ...          # sync loop
    async def arun(self, prompt, **kwargs) -> ResponseMem: ...   # async loop
    async def astream(self, prompt, **kwargs) -> AsyncIterator[ResponseChunk]: ...
```

Loop logic: invoke → if tool_calls → check permission → run tools → feed results back (no new user message) → repeat until no tool calls or max_turns hit.

### 3. Continuation call — internal to `Agent`

Agent never appends a "Continue." message. After tool results it calls the underlying adapter directly via `model.llm.invoke(prompt=None)` — adapters must accept `prompt=None` and skip appending a user turn when it is `None`.

### 4. `ResponseChunk` fields — `core/base/base_model.py`

```python
@dataclass
class ResponseChunk:
    text: str
    is_final: bool = False
    function_call: Optional[Any] = None
    finish_reason: Optional[str] = None   # "stop" | "tool_calls" | "max_tokens" | None
    input_tokens: int = 0
    output_tokens: int = 0
```

Adapters populate these on the final chunk where the provider exposes them.

### 5. Native async — per-adapter

| Adapter            | Change                                                                                |
| ------------------ | ------------------------------------------------------------------------------------- |
| `claude`           | `AsyncAnthropic`; `async with client.messages.stream()`                               |
| `gemini`           | `client.aio.models.generate_content_stream()` (google-genai async)                    |
| `openai_`          | `AsyncOpenAI`; `async for chunk in await client.chat.completions.create(stream=True)` |
| `universal_openai` | Same as openai\_                                                                      |
| `deepseek`         | Same pattern (AsyncOpenAI with base_url)                                              |
| `xai`              | Already uses `AsyncClient`; remove remaining threading                                |
| `together`         | Keep threading — Together SDK has no stable async API; add cancellation check         |
| `vertex`           | Keep threading — vertexai async support unclear; add cancellation check               |

Remove all `threading.Thread` + `asyncio.Queue` boilerplate from the adapters listed above. Each adapter's `astream` becomes a clean `async for` over the provider's native async iterator.

### 6. Cancellation — all adapters + `core/_runner.py`

Add `abort: Optional[asyncio.Event] = None` to every `astream` signature. In `_runner.py`:

```python
async for chunk in self.llm.astream(..., abort=abort):
    if abort and abort.is_set():
        return
    yield chunk
```

For native-async adapters, `asyncio.CancelledError` also propagates naturally. For threading adapters (Together, Vertex), check `abort.is_set()` in the queue loop between each item.

### 7. Scoped `ToolRegistry` — `core/tool.py`

```python
class ToolRegistry:
    def __init__(self): self._tools: Dict[str, Tool] = {}; self._callbacks = []
    def register(self, tool: Tool) -> None: ...
    def autotool(self, description="", ...): ...   # scoped decorator
    def all(self) -> Dict[str, Tool]: ...

default_registry = ToolRegistry()
TOOL_REGISTRY = default_registry._tools   # backward-compat alias
```

`Model.__init__` accepts `tool_registry: Optional[ToolRegistry] = None`; falls back to `default_registry`.

### 8. History truncation warning — `core/base/memory.py`

```python
def __init__(self, ..., on_truncate: Optional[Callable] = None): ...

def _size_update(self):
    if len(self.MemoryList) > self.size:
        dropped = self.MemoryList[: len(self.MemoryList) - self.size]
        self.MemoryList = self.MemoryList[-self.size:]
        if self.on_truncate:
            self.on_truncate(dropped)
        else:
            import warnings
            warnings.warn(f"History truncated: {len(dropped)} entries dropped")
```

### 9. Session — `agent/session.py`

```python
@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid4().hex)
    history: History = field(default_factory=History)
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Literal["idle", "running", "done"] = "idle"
```

`Agent.arun` sets `state` appropriately and raises if called while `running` (concurrency guard).

### 10. Permission system — `agent/agent.py`

`Agent(permission=fn)` where `fn(tool_name: str, args: dict) -> bool` (sync or async). Called before each tool execution in the loop. If returns `False`, the tool is skipped and a synthetic "permission denied" result is injected.

### 11. Event bus — `agent/events.py` + `agent/agent.py`

```python
class EventBus:
    def __init__(self): self._handlers: Dict[str, List[Callable]] = {}
    def on(self, event: str, fn: Callable) -> None: ...
    def emit(self, event: str, *args) -> None: ...
```

Events emitted by `Agent`: `text_delta`, `tool_start`, `tool_end`, `step_end`. `Agent` accepts `events: Optional[EventBus]`. `model.on(event, fn)` is a convenience proxy.

### 12. `automem` continuation entries — `core/_runner.py`

In `invoke`/`ainvoke`/`astream`, only add `UserMem` when `prompt is not None`. Continuation calls (from `Agent`) pass `prompt=None`.

### 13. `tool_results` atomicity — `core/base/memory.py`

```python
def complete_tool_call(mem: ResponseMem, results: list, history: Optional[History] = None) -> None:
    """Set tool_results on mem; if history given, ensures mem is in it."""
    mem.tool_results = results
    if history and mem not in history.MemoryList:
        history.add(mem)
```

---

## Line-count compliance

Any adapter file still exceeding 200 lines after the async refactor must extract its `to_messages()` / `to_contents()` history-serialisation logic into a sibling `helpers.py` (Gemini already has this pattern).

---

## Verification

1. `python3 -c "from synaptic import Model, Agent, Session, ToolRegistry, Provider; print('imports OK')`
2. Run existing tests if any under `tests/`
3. Manual smoke test: `Agent(model).arun("list files")` with a filesystem tool — confirm multi-turn loop terminates and history is correct
4. Check `model.history = new_history` syncs to `model.llm.history` (issue 1)
5. Test abort: set `abort_event` mid-stream and confirm generator exits cleanly
6. Confirm no adapter file exceeds 200 lines after all changes
