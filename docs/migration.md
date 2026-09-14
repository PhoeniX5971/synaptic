# Migration notes

## First hardening tranche

- ResponseMem, ResponseChunk and TurnComplete token counts default to None when
  unavailable. Use `value or 0` only when deliberately aggregating unknown as zero.
  Memories also expose finish_reason and provider extensions.
- Model.bind_tools([tool, tool]) deduplicates. A different explicit definition
  with the same name raises; pass replace=True to update it. unbind_tools(*names)
  removes local bindings. A registry-owned tool is removed with registry.unregister;
  removing a local override can reveal a same-name registry tool. Read Model.tools
  as a snapshot; use binding/registry methods to change provider schemas.
- Retry defaults to ConnectionError/TimeoutError and never restarts after any
  yielded chunk. Model autorun must be False to retry; unknown producers require
  retry_safe=True, asserting they cannot execute effects before returning/yielding.
  No generic restart/reset protocol is offered. `@retry` remains caller-controlled
  and can repeat effects; configure its exception filter and ensure repetition is safe.
- Async execution resolves Future/custom awaitable results. Sync helpers reject
  async results and close returned coroutines; they do not cancel caller-owned
  Futures or claim a rejected awaitable has not already performed work.
- Historical History.MemoryList/memoryList became entries/memory_list before this
  tranche; the recovered tests now reflect that prior migration.

## MCP results and lifecycle

MCP tool wrappers now return a JSON envelope, not a string or bare structured value:

```python
from synaptic.mcp import extract_text, legacy_result

result = await tool.run(text="hello")
print(result["structured_content"])
print(result["content"])  # all typed media/text/resource blocks
print(extract_text(result))
old_shape = legacy_result(result)  # explicitly lossy compatibility path
```

Envelope version 1 also carries has_structured_content, is_error, metadata,
extensions and provenance. `normalize_result(raw_sdk_result)` is available for
MCPSession.call_tool users. Tool failure is `is_error=True`; call-level protocol
errors raise MCPProtocolError with code/data; transport failures raise
MCPTransportError. A transport timeout/cancellation does not prove no tool effect
occurred. These operations never automatically replay a tool call.

Failed sessions require `await session.reconnect()` or close then connect.
`close()` is idempotent. A shutdown timeout leaves state `closing`, preventing
replacement until the old task actually exits. Do not concurrently share one
session across event loops. Periodic ping detects idle transport death; detection
is bounded by health_interval + request_timeout, not instantaneous.

`MCPSession(..., discovery_limits=DiscoveryLimits(...), connect_timeout=...,
shutdown_timeout=..., health_interval=..., request_timeout=...)` configures limits.
Discovery consumes all pages and returns a copy; limit/validation/timeout failures
preserve the previous catalog. MCPHub accepts existing server configs but its
registry refresh/removal semantics are still under T-006 review.

Install the MCP extra (`mcp>=1.30,<2`). Only 1.30.0 is measured so far; 2.x is
excluded pending an explicit API/protocol migration.
