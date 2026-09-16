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
