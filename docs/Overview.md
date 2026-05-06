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

`Flow` and `Pipeline` are small workflow helpers for deterministic app logic
outside the model loop.

## Public Imports

```python
from synaptic import (
    Agent,
    EventBus,
    History,
    Model,
    Provider,
    Session,
    Tool,
    ToolRegistry,
    autotool,
)
```

## When To Use What

- Use `Model.invoke()` for one request.
- Use `Model.astream()` for streaming text.
- Use `Model(autorun=True)` for one-turn tool execution.
- Use `Agent.run()` / `Agent.arun()` for multi-turn tool workflows.
- Use `ToolRegistry` to isolate tools between sessions or tests.
- Use `History(size=N)` to control context size.
- Use `Flow` or `Pipeline` for predictable non-agent workflows.
