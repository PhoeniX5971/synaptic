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

With a schema (a dict or a Pydantic-style class exposing `model_json_schema`), Synaptic
sends an OpenAI `json_schema` response format so the output is constrained to it. Pass
`enforce_schema=False` to request a plain JSON object instead.

## Streaming

```python
async for chunk in model.astream("Stream this"):
    print(chunk.text, end="")
```

The OpenAI adapter uses the async OpenAI client for streaming.
