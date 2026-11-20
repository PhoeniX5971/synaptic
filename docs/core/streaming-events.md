# Streaming And Events

Synaptic streaming uses async iterators that yield `ResponseChunk`.

## ResponseChunk

```python
ResponseChunk(
    text="",
    is_final=False,
    function_call=None,
    finish_reason=None,
    input_tokens=0,
    output_tokens=0,
)
```

- `text`: text delta or accumulated final text.
- `is_final`: true on the final chunk.
- `function_call`: optional `ToolCall`.
- `finish_reason`: provider finish reason when available.
- `input_tokens` / `output_tokens`: token counts when available.

## Model Streaming

```python
async for chunk in model.astream("Write a status update"):
    print(chunk.text, end="")
```

## Abort Streaming

```python
import asyncio

abort = asyncio.Event()

async for chunk in model.astream("Long task", abort=abort):
    if should_stop():
        abort.set()
```

Native async adapters stop cooperatively. Together and Vertex use background
threads because their SDKs do not provide the same async streaming surface; they
also check the abort event.

## EventBus

```python
from synaptic import EventBus

events = EventBus()
events.on("text_delta", lambda text: print(text, end=""))
events.on("tool_start", lambda call: print("tool", call.name))
```

Async handlers are supported by `aemit`, which `Agent.arun` uses.

## Agent Events

- `text_delta`: streamed text from `Agent.astream`.
- `tool_start`: before a tool runs.
- `tool_end`: after a tool result or error. The observer's `result` copy carries
  `latency_ms`: executor time only, excluding policy and observer time (0 when denied).
- `step_end`: after each model response in an agent loop; `response.latency_ms` is the
  model-call duration.

## Tool Latency

```python
agent.on("tool_end", lambda call, result: print(call.name, result["latency_ms"], "ms"))
```

The same value is journaled per call ID in `response.tool_outcomes[call_id]["latency_ms"]`
(also for `Model(autorun=True)`), on the `ToolCallResult` signal and in the `agent.tool`
log event. It is never added to `tool_results`, so it is not sent to the provider.
