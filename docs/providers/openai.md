# Provider: OpenAI

Use OpenAI through `Model(provider=Provider.OPENAI, ...)`.

```python
from synaptic import Model, Provider

model = Model(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    api_key="...",
    instructions="Answer clearly.",
)

res = model.invoke("Hello")
print(res.message)
```

## Tools

```python
model = Model(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    api_key="...",
    tools=[tool],
    autorun=True,
)
```

OpenAI tools are sent as chat completion function tools. Tool calls are parsed
into `ToolCall` objects and stored on `ResponseMem.tool_calls`.

## Structured JSON

```python
from synaptic import ResponseFormat

model = Model(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    api_key="...",
    response_format=ResponseFormat.JSON,
    response_schema=MyPydanticModel,
)
```

When a Pydantic-style schema exposes `model_json_schema`, Synaptic sends an
OpenAI JSON schema response format. Otherwise it requests a JSON object.

## Streaming

```python
async for chunk in model.astream("Stream this"):
    print(chunk.text, end="")
```

The OpenAI adapter uses the async OpenAI client for streaming.
