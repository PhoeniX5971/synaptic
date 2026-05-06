# Provider: Gemini

Use Gemini through `Model(provider=Provider.GEMINI, ...)`.

```python
from synaptic import Model, Provider

model = Model(
    provider=Provider.GEMINI,
    model="gemini-2.5-flash",
    api_key="...",
)

res = model.invoke("Hello")
print(res.message)
```

## Tools

Gemini tools are converted to `google.genai.types.Tool` objects.

```python
model = Model(
    provider=Provider.GEMINI,
    model="gemini-2.5-flash",
    api_key="...",
    tools=[tool],
)
```

## Images And Audio

`invoke`, `ainvoke`, and `astream` accept `images=` and `audio=` lists when the
provider adapter supports those inputs.

```python
res = model.invoke(
    "Describe this image",
    images=["./diagram.png"],
)
```

Local files are loaded as inline provider parts. URLs are downloaded and sent as
inline data.

## Structured JSON

```python
model = Model(
    provider=Provider.GEMINI,
    model="gemini-2.5-flash",
    api_key="...",
    response_format=ResponseFormat.JSON,
    response_schema=MySchema,
)
```

Structured output disables Gemini tool declarations for that request.

## Streaming

Gemini streaming uses `client.aio.models.generate_content_stream`.
