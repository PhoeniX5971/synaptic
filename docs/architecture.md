# Architecture

Model is the provider-neutral request boundary. `_factory.py` lazily imports only the
selected adapter. Adapters own SDK clients and wire conversions; their capabilities
make unsupported modes/media explicit. `_runner.py` handles invoke bookkeeping,
`_model_stream.py` owns streaming cancellation/history, and `_completion.py` assembles
usage, final text, finish metadata and deduplicated completed call IDs. Sync, async
aggregation and streams share the same completion records.

ToolRegistry owns versioned definitions and exact catalog ownership. Registry reads
are protected snapshots; Tool definitions are immutable. A request captures its tool
view before generation. `Tool.prepare` resolves defaults and validates a captured
Invocation. Agent authorizes that effective invocation; Executor owns the actual effect.
LocalExecutor is useful by default, while proposal-only Model operations plus correlated
external results allow applications to own execution and persistence completely.

Agent owns Session history, turn budgets and policy. It adds each user message once,
prepares the turn's tool view, disables Model autorun and records tool results against
stable IDs. Signal mode converts text DSL into proposals. EventBus observers receive
copies and have a separate failure policy. Session outcome distinguishes completion,
failure, cancellation and exhausted budgets. Model/History operations reject concurrent
owners; separate instances are the unit of concurrent isolation.

History stores typed content references, completion metadata and complete tool exchanges.
Window truncation drops whole entries, optionally using an application token counter.
Version-1 dictionaries allow optional external storage. Repeated-result compaction is
explicitly lossy. Persistence, desktop permissions, application grants and PC scheduling
remain consumer responsibilities.

MCPHub owns bounded named sessions and registration transactions. Canonical identities
are separate from provider aliases; each hub's exact registrations are independently
removable. It prepares the whole catalog before publishing registry and loaded state,
and notification failure cannot roll it back. `_hub_runtime.py` bounds workers,
connections and operation time, drains active calls, evicts idle connections on access,
and applies reconnect backoff without replaying a failed call.

MCPSession owns one asyncio task per connection attempt so SDK/AnyIO transport contexts
enter and exit in the same task. `_connection.py` handles lifecycle/readiness, `_transport.py`
opens official SDK transports, `_discovery.py` bounds and validates paginated catalogs,
and `results.py` returns the lossless JSON envelope. Every dispatched call captures the
attempt/session/task that may be invalidated by its failure. Late requests cannot
invalidate newer owners. No shared credential-based pool or discovery cache exists.

ThreadScheduler offers explicit bounded offloading for synchronous functions in
LocalExecutor, Flow/Pipeline and EventBus. Cancellation releases the awaiter, not the
worker's capacity, until actual completion. Provider async APIs remain native except
Together's bounded synchronous iterator bridge. Retry repeats only eligible generation
before visible output, with finite backoff/deadlines; uncertain effects are replay barriers.

DiagnosticBus uses context-local bus selection and correlation, suppresses payloads by
default, and redacts before every sink. QueuedSink bounds delivery and reports loss/failure;
JSONLWriter rotates retained files. These diagnostics are best-effort. Applications needing
a durable effect journal must implement it at their executor/storage boundary.

Public semantics and limits: [contracts](contracts.md). Provider differences:
[capability matrix](provider-matrix.md). Executed evidence: [qualification](qualification.md).

## Readiness follow-up: execution evidence and admission

ResponseMem is the canonical per-turn execution record. Model/Agent attach it before
side effects, then a response-scoped ExecutionState publishes receipts and call-ID states
incrementally into existing serializable extensions. Event and diagnostic observers are
downstream of receipt publication. Exceptions preserve the same response; incomplete
history is rejected before provider continuation until the consumer reconciles it.

Invocation carries the cancellation/admission gate across the executor boundary.
Local admission occurs immediately before calling the captured function, including inside
a scheduled worker. Cooperative external executors use the same gate after their own
queues; opaque executors are conservatively considered started at handoff. Started work
can remain uncertain after cancellation. This is an in-memory evidence boundary; durable
storage, remote idempotency and authoritative reconciliation remain consumer-owned.

## Integration follow-up: independent retention and distribution gates

Retry's context-local generation-attempt owner routes only invisible effect-free failures
to History.failed_attempts. Partial output, proposals and admitted effects retain normal
execution barriers; history ownership spans the retry/backoff operation.

History.entries is the provider context projection. History.recovery separately retains
tool exchanges and incomplete responses across context truncation. Framework result
publication, reconciliation and provider entry all recheck budgets; unresolved recovery
blocks continuation regardless of context membership. Serialization preserves recovery
and failed-attempt evidence with explicit new-reader compatibility rules.

The source gate is complemented by installed-wheel consumer typing tests. py.typed,
explicit exports and preserved decorator signatures expose useful public types without
requiring optional SDKs in core-only installations. Tracked CI invokes the frozen quality
and release scripts; actual remote-run evidence remains separate from local qualification.
