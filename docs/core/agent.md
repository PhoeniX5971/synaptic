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
    max_parallel_tools=1,
    ledger=None,
    context=None,
)
```

- `max_parallel_tools`: concurrency limit for tools marked parallel-safe (see tool guide).
- `ledger`: a `Ledger` that records every run (see ledger guide).
- `context`: function choosing what each model call sees (see Context below).

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
agent.on("tool_end", lambda call, result: print(result, result["latency_ms"]))
agent.on("step_end", lambda response: print(response.message))
```

Streaming also emits `text_delta`.

## Streaming

`Agent.astream()` streams one model turn and emits text events. Use `arun()` for
the full multi-turn continuation loop.

## Task State

`agent.task` (the same dict as `session.task`) holds facts about the work that should
not depend on the conversation: goals, completed steps, IDs. Tools update it through a
closure or bound method; it is saved with the session.

```python
@autotool("Mark a step complete")
def complete(step: str) -> str:
    agent.task.setdefault("done", []).append(step)
    return "recorded"
```

Keep values JSON-compatible. Do not confuse it with `session.state`, which is the run
lifecycle (`idle`, `running`, `done`, ...).

## Context

By default each model call sees the history window. Pass `context` to decide instead:

```python
def compile_context(entries, task):
    brief = UserMem(f"Goal: {task['goal']}. Done: {task.get('done', [])}", None)
    return [brief, *entries[-6:]]

agent = Agent(model, context=compile_context)
```

The function runs for every provider request and returns the memories to send. The
stored history is never changed, so nothing is lost. Keep it fast and free of side
effects. A `ResponseMem` holds its tool calls and results together, so dropping or
keeping whole entries never splits a tool exchange; ordering and a valid first message
for your provider are your responsibility. `history.compiler = fn` does the same for a
bare `Model`.

## Saving and resuming

```python
saved = json.dumps(agent.session.to_dict())
...
session = Session.from_dict(json.loads(saved))
Agent(model, tools=tools, session=session).run(None)   # continue without a new prompt
```

`run(None)` continues from the stored history. If the saved run stopped with a tool
call whose outcome is unknown, continuing raises until you reconcile it with
`complete_tool_call`. Where the saved session is stored is up to you.

## Handoff

```python
agent.on("tool_end", lambda call, result: agent.handoff(stronger_model))
```

`agent.handoff(model)` moves the run to another model at the next turn boundary: same
session, history, tools and task, no new user prompt, and a stream being consumed simply
continues. The response in progress finishes on the current model. The target keeps its
own instructions, or inherits the agent's if it has none. It must not be in use
elsewhere and must use the same signal mode.

## Subagents

```python
researcher = Agent(research_model, tools=[search])
lead = Agent(model, tools=[researcher.as_tool("research", "Research a topic in depth")])
```

The delegate runs its own loop with its own session and turn budget and returns its
final message as the tool result. Cancelling the lead cancels the delegate. Give the
delegate its own `Model` instance. Reusing one agent keeps its memory between
delegations and runs them one at a time; pass a factory for fresh, parallel delegates:

```python
from synaptic import subagent
tool = subagent("research", "Research a topic", lambda: Agent(make_model(), tools=[search]))
```

Use `sync=True` when the lead runs with `Agent.run`.
