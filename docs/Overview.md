# Overview

Synaptic turns provider SDKs into one simple application API.

## Architecture

`Model` is the lowest user-facing layer. It wraps a provider adapter and knows
how to invoke, stream, bind tools, and write memory.

`Agent` sits above `Model`. It runs a multi-turn loop: model call, tool calls,
tool results, continuation call, final answer.

`History` stores conversation entries. Provider adapters serialize it into the
format each SDK expects.

`Tool` and `ToolRegistry` define callable work the model may request. A tool can
be bound directly to a model or scoped through a registry.

`EventBus` broadcasts agent events such as streamed text and tool start/end.

`Ledger` records what a run did, in order, for inspection, storage and replay.

`Session` carries the conversation plus `task`, a dict for facts about the work that
should not depend on the conversation; it serializes for saving and resuming.

An `Agent` can also compile its own context per request, hand a live run to another
model, delegate to other agents as tools, and run parallel-safe tools concurrently.

Extensions add opt-in behaviour through one seam: `FinalAnswer` ends a run with a
validated structured result, `Todos` keeps a plan whose required items must be done
before finishing, and `Skills` loads instructions and tools only when a task needs them.

`Flow` and `Pipeline` are small workflow helpers for deterministic app logic
outside the model loop.

## Public Imports

```python
from synaptic import (
    Agent,
    EventBus,
    History,
    Ledger,
    Model,
    Provider,
    Session,
    Tool,
    ToolRegistry,
    autotool,
    subagent,
)
```

## When To Use What

- Use `Model.invoke()` for one request.
- Use `Model.astream()` for streaming text.
- Use `Model(autorun=True)` for one-turn tool execution.
- Use `Agent.run()` / `Agent.arun()` for multi-turn tool workflows, and
  `Agent.astream()` for the same run with chunks as they arrive.
- Use `Agent(extensions=[...])` with `FinalAnswer`, `Todos` or `Skills` for structured
  results, required steps and on-demand capabilities.
- Use `Agent(retry=RetryPolicy(...))` to repeat model calls that fail before any output.
- Use `Ledger` to record, store, fork or replay runs.
- Use `agent.task` for facts about the work, `Agent(context=...)` to choose what the
  model sees, `agent.handoff(model)` to switch models mid-run, and `agent.as_tool(...)`
  to delegate to another agent.
- Use `ToolRegistry` to isolate tools between sessions or tests.
- Use `History(size=N)` to control context size.
- Use `Flow` or `Pipeline` for predictable non-agent workflows.
