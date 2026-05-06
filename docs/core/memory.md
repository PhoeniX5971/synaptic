# Core: Memory

Memory stores conversation state in provider-neutral Python objects.

## Memory Types

`Memory` is the base class:

```python
Memory(message="hello", created=created, role="user")
```

`UserMem` defaults to role `user`:

```python
UserMem("hello", created=created)
```

`ResponseMem` represents assistant output:

```python
ResponseMem(
    message="answer",
    created=created,
    tool_calls=[],
    tool_results=[],
)
```

## History

`History` is an ordered sliding window:

```python
from synaptic import History

history = History(size=10)
history.add(UserMem("hello", created=created))
```

When more than `size` entries exist, old entries are dropped.

## Truncation

Use `on_truncate` when dropped context matters:

```python
def save_dropped(memories):
    archive.extend(memories)

history = History(size=20, on_truncate=save_dropped)
```

Without `on_truncate`, Synaptic emits a warning when history truncates.

## Tool Calls

`ResponseMem.tool_calls` stores parsed provider function calls.

```python
names = response.list_tool_calls()
call = response.get_tool_call("search")
```

`get_tool_call` raises `IndexError` if the name is missing.

## Completing Tool Calls

Agents use `complete_tool_call` to attach results atomically:

```python
from synaptic import complete_tool_call

complete_tool_call(response, [{"name": "search", "result": "..."}], history)
```

If a history is passed and the response is not already present, it is added.

## Prompt Continuation

When an agent continues after tools run, it calls the provider with
`prompt=None`. That means no new `UserMem` is added for the continuation.
