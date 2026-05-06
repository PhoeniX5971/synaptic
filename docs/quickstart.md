# Quickstart

This guide builds a small Synaptic app with a model, a tool, memory, and an
agent.

## 1. Create A Model

```python
from synaptic import Model, Provider

model = Model(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    api_key="...",
    instructions="Be direct and useful.",
)

res = model.invoke("Say hello in one sentence")
print(res.message)
```

`Provider` selects the adapter. `model` is the provider's model id.

## 2. Add A Tool

```python
from synaptic import autotool

@autotool("Add two integers")
def add(a: int, b: int) -> int:
    return a + b
```

Pass the tool to `Model`:

```python
model = Model(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    api_key="...",
    tools=[add],
    autorun=True,
)

res = model.invoke("Use a tool to add 7 and 8")
print(res.tool_results)
```

`autorun=True` executes returned tool calls for that one model turn.

## 3. Use An Agent

```python
from synaptic import Agent

agent = Agent(model, max_turns=5)
res = agent.run("Add 7 and 8, then explain the answer")
print(res.message)
```

`Agent` continues after tools run, so the final answer can use tool results.

## 4. Stream Text

```python
async for chunk in model.astream("Write a short poem"):
    print(chunk.text, end="")
```

Each chunk is a `ResponseChunk` with `text`, `is_final`, optional
`function_call`, and optional token metadata.

## 5. Keep Sessions Separate

```python
from synaptic import Session, ToolRegistry

registry = ToolRegistry()
session = Session()
model = Model(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    api_key="...",
    tool_registry=registry,
)
agent = Agent(model, session=session)
```

Use one `Session` per conversation and one `ToolRegistry` when tool scope must
be isolated.
