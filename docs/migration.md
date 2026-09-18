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

## Completed follow-up contracts

- Install only required provider extras; `pip install -e '.[all]'` installs all.
  Core now requires Pydantic 2 and jsonschema, with Python >=3.11. SDK imports are lazy;
  missing support names the required extra. `.env` loading belongs to the application.
  The existing LICENSE text is included as package metadata without relicensing.
- Models now have isolated registries. Explicit `tools=[...]`, `tool_registry=...`,
  or `legacy_registry=True` replace reliance on implicit global decorators.
  Agent tools must be registered with the Agent. Close disposable subscriptions
  and call `await model.aclose()` when async clients were used.
- Tool definitions/defaults/metadata are snapshots. Replace definitions rather than
  assigning fields. `registry.all()` is read-only. Caller arguments override defaults;
  policy sees validated effective arguments. Pydantic coercion is explicit behavior.
- Use injectable Executors or Model.propose/apropose for externally owned effects.
  Streaming proposals use autorun=False. ToolCall.id is native or generated and stable;
  supply_results requires complete, unique correlation. Partial deltas cannot execute.
- MCP unload uses exact ownership, and aliases containing separators/invalid characters
  or exceeding 64 characters change to bounded hashes. Resolve via hub.resolve_alias.
  load_all now returns per-server exceptions alongside successes. Routing registrations
  are hub-owned. Unload does not revoke captured handles; remove drains and invalidates.
- EventBus defaults to isolated observer errors; choose errors="raise" for fail-fast
  behavior. Observer callbacks receive copies. Agent.session.outcome distinguishes
  cancellation, failure and budget exhaustion; reset retains conversation history.
- Finish reasons use portable categories; inspect extensions.native_finish_reason for
  differing native values. xAI raw audio now fails explicitly. Read provider-matrix.md
  for actual supported modalities and differences in structured output.
- ContentBlock and version-1 history serialization preserve media and call correlation.
  Optional token budgets need an application-supplied counter. Lossy duplicate-result
  compaction stays opt-in. Interrupted streamed memories are marked incomplete.
- Retry includes transient SDK/status classification, retry-after, exponential bounded
  jitter and total deadlines. Sync deadlines cannot preempt a blocking call. Autorun
  and first visible output remain replay barriers; generic @retry remains caller-owned.
- Diagnostics default to metadata. Use DiagnosticBus(payloads=True, secrets=...,
  redact=...) for intentional payload collection. Redaction precedes every sink.
  scope(session_id=..., run_id=...) isolates correlation; QueuedSink bounds delivery,
  exposes dropped/failures counters, and drains on close. JSONLWriter rotates files.

The earlier statement that hub refresh remained under review is historical and is
superseded by these contracts. See [runtime contracts](contracts.md) for lifecycle
semantics and [qualification](qualification.md) for exact measured limits.

## Control readiness follow-up

Model autorun now stops and raises on tool/executor/output-validation failure. Earlier
receipts remain in history when automem is enabled, and in `model.last_response` otherwise.
Agent preserves the same evidence for failures/cancellation, including strict observer
errors. Tool result records now carry call IDs on all Model autorun paths.

Use `response.tool_outcomes` to distinguish succeeded, denied, not_started, started,
uncertain and reconciled calls. History v1 preserves these in extensions. An incomplete
response blocks provider continuation even if all receipts exist; explicitly acknowledge
with complete_tool_call/supply_results, retaining known receipts unchanged. Successful or
uncertain calls cannot be replayed by execution helpers. This does not provide durable
exactly-once execution across process failure.

Pass `abort` on sync, async or streaming Model/Agent calls. Custom delayed executors can
opt into `manages_admission=True` and use Invocation.check_cancelled/begin_effect after
queue admission; existing opaque executors retain conservative uncertain-effect semantics.
See contracts.md before implementing a custom executor.

## Control integration review (T-011)

- Retry + Model + automem now retries invisible effect-free generation failures without
  duplicating prompts. Failed-attempt records live in history.failed_attempts (also with
  automem disabled). Partial output and autorun effects remain replay barriers.
- History.entries is now actively rechecked as bounded provider context after result
  growth and before submission. An oversized exchange can disappear from context while
  remaining in history.recovery. Inspect recovery for receipts and reconcile pending
  entries there; dropping context no longer permits bypassing incomplete execution.
- History v1 adds optional recovery/failed_attempts fields. New readers accept old records;
  older software may discard the new sections and must not round-trip these snapshots.
  Reattach token counters/budget policies after restoration. Retention is consumer-owned.
- Contradictory IDs, mixed tagged/positional inputs and ambiguous partial positional
  records now raise before mutation, including on restoration. Valid complete untagged
  positional lists remain supported. Known receipts cannot be replaced by changing format.
- Wheel/sdist now include py.typed. Model/Agent invocation signatures and Model stream/
  proposal returns remain typed after ownership decoration. Dynamic SDK payloads and
  generic Retry targets still have intentionally limited static typing.

Run scripts/check_quality.py and scripts/check_release.py under the frozen development
Python. The install harness now invokes the development host's mypy against each selected
isolated target, so run it with that host rather than a bare core-only Python interpreter.
