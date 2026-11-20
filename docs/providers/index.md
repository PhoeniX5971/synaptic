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

## Tool results

Synaptic keeps one tool-result shape internally and each adapter sends it in the
provider's own form: the call ID travels in `tool_call_id` (OpenAI-compatible, xAI),
`tool_use_id` (Claude) or the function-response ID (Gemini). Failures use the provider's
error marking where it has one. The exact table is in [runtime contracts](../contracts.md).

## Enforced schemas on OpenAI-compatible servers

`Provider.UNIVERSAL_OPENAI` and `Provider.DEEPSEEK` send `response_format: json_object`
by default, because many compatible servers reject `json_schema`. If yours supports it
(vLLM with guided decoding, for example), opt in so the output must match your schema:

```python
model = Model(
    provider=Provider.UNIVERSAL_OPENAI,
    model="my-model",
    base_url="http://localhost:8000/v1",
    response_format=ResponseFormat.JSON,
    response_schema=AskDraft,
    enforce_schema=True,
)
```

`enforce_schema` only affects these Chat Completions adapters; `Provider.OPENAI` enforces
by default. Whether the server honours the schema is up to the server.
