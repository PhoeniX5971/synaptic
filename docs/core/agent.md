# Core: Agent

`Agent` wraps a `Model` with a multi-turn tool loop.

## Why Use Agent

`Model(autorun=True)` can execute tools for one model response. `Agent` goes
further: after tools run, it calls the model again with the tool results in
history, so the model can produce a final answer.

## Constructor

```python
Agent(
    model,
    session=None,
    max_turns=10,
    permission=None,
    events=None,
)
```

- `model`: configured `Model`.
- `session`: `Session` with isolated history and state.
- `max_turns`: maximum model/tool cycles.
- `permission`: callback deciding whether a tool may run.
- `events`: `EventBus` instance.

## Run

```python
agent = Agent(model)
res = agent.run("Use tools if needed")
print(res.message)
```

Async:

```python
res = await agent.arun("Use tools if needed")
```

## Permission Callback

```python
def allow(tool_name: str, args: dict) -> bool:
    return tool_name != "delete_file"

agent = Agent(model, permission=allow)
```

Async permission callbacks are supported by `arun`.

If permission returns `False`, Synaptic records:

```python
{"name": tool_name, "error": "Permission denied"}
```

## Sessions

```python
from synaptic import Session

session = Session(metadata={"user_id": "u_123"})
agent = Agent(model, session=session)
```

The agent sets `session.state` to `running` during work and `done` afterward.
It raises if the same session is already running.

## Events

```python
agent.on("tool_start", lambda call: print(call.name))
agent.on("tool_end", lambda call, result: print(result))
agent.on("step_end", lambda response: print(response.message))
```

Streaming also emits `text_delta`.

## Streaming

`Agent.astream()` streams one model turn and emits text events. Use `arun()` for
the full multi-turn continuation loop.
