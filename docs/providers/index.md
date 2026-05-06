# Providers

Synaptic exposes providers through the `Provider` enum and `Model`.

## Provider Values

```python
from synaptic import Provider
```

- `Provider.CLAUDE`
- `Provider.OPENAI`
- `Provider.GEMINI`
- `Provider.VERTEX`
- `Provider.DEEPSEEK`
- `Provider.TOGETHER`
- `Provider.XAI`
- `Provider.UNIVERSAL_OPENAI`

## Common Pattern

```python
from synaptic import Model, Provider

model = Model(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    api_key="...",
)
```

All providers return `ResponseMem` from `invoke` and `ainvoke`, and
`ResponseChunk` from `astream`.

## Provider Notes

- OpenAI, DeepSeek, and Universal use OpenAI-compatible chat APIs.
- Claude uses Anthropic Messages.
- Gemini uses Google GenAI.
- Vertex uses Vertex AI generative models.
- XAI uses xAI SDK clients.
- Together uses the Together SDK.

## Async Streaming

OpenAI, DeepSeek, Universal, Claude, Gemini, and XAI use native async streaming.
Together and Vertex stream through a background thread with abort checks because
their sync SDK surface is the stable path used here.

## Universal OpenAI

Use `Provider.UNIVERSAL_OPENAI` for local or hosted OpenAI-compatible servers.

```python
model = Model(
    provider=Provider.UNIVERSAL_OPENAI,
    model="llama",
    base_url="http://localhost:11434/v1",
    api_key="none",
)
```
