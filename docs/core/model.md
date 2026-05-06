# Core: Model

`Model` is the provider-neutral interface for a single LLM configuration.

## Constructor

```python
Model(
    provider,
    model,
    temperature=0.8,
    api_key="",
    max_tokens=1024,
    tools=None,
    history=None,
    autorun=False,
    automem=False,
    blacklist=None,
    location=None,
    project=None,
    instructions="",
    response_format=ResponseFormat.NONE,
    response_schema=None,
    base_url=None,
    tool_registry=None,
)
```

## Important Options

- `provider`: a `Provider` enum value.
- `model`: provider model id.
- `tools`: tools bound only to this model.
- `tool_registry`: scoped registry used by the adapter.
- `history`: `History` object used as provider context.
- `autorun`: execute tool calls after a model response.
- `automem`: write user and assistant turns to history.
- `instructions`: system or developer guidance sent to the provider.
- `response_format`: `ResponseFormat.NONE` or `ResponseFormat.JSON`.
- `response_schema`: schema used for structured JSON responses.
- `blacklist`: tool names that must not be executed.
- `location` / `project`: Vertex provider configuration.
- `base_url`: Universal OpenAI-compatible endpoint.

## Invocation

```python
res = model.invoke("Hello")
print(res.message)
```

`invoke` runs one synchronous call. `ainvoke` has the same behavior but is
awaitable.

```python
res = await model.ainvoke("Hello")
```

## Streaming

```python
async for chunk in model.astream("Tell me a story"):
    print(chunk.text, end="")
```

Pass an `asyncio.Event` as `abort=` to stop a stream.

## Memory

When `automem=True`, Synaptic stores the user prompt and assistant response:

```python
from synaptic import History

history = History(size=20)
model = Model(..., history=history, automem=True)
model.invoke("Remember this")
```

`model.history = new_history` also updates the underlying provider adapter.

## Tools

`autorun=True` runs tools for one response:

```python
model = Model(..., tools=[tool], autorun=True)
res = model.invoke("Use the tool")
print(res.tool_results)
```

For multi-turn tool use, prefer `Agent`.

## Continuation

Adapters accept `prompt=None`. Agents use this after tool results so the next
provider call continues from history without adding a fake user message.
