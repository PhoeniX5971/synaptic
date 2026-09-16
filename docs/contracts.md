# Runtime contracts

These contracts apply to the hardening follow-up. [Qualification](qualification.md)
separates local fixtures from live services; [migration](migration.md) lists changes.

## Model, tools and execution

A Model owns a scoped ToolRegistry and History by default. Share a registry explicitly
when appropriate; use `legacy_registry=True` only for the process-global compatibility
registry. An Agent owns a Session and its authorized tool set. Pass its tools or
registry explicitly. Model/History reject concurrent operations from different tasks
or threads. An Agent may make nested model calls in its owning task. Public field
mutation during a run is unsupported; use separate instances for concurrent sessions.

Tool definitions are immutable, and registry snapshots are read-only. Replace a
registration to change a definition. Explicit model bindings override registry tools;
conflicting explicit bindings require `replace=True`. A turn captures definitions
before generation, so a subsequent catalog change cannot rebind an already proposed
call to another function. MCP aliases resolve through `hub.resolve_alias(alias)` to
`ToolIdentity(server, tool)`; simple names retain `server__tool`, otherwise a bounded
SHA-256 alias is used. Collisions fail before publication.

`autotool` uses Pydantic 2, including resolved annotations, nested models/collections,
Optional/union, Literal and enums. Unsupported annotations and variadic/positional-only
signatures fail explicitly. Unannotated values use Any. Pydantic's normal conversion
rules apply; authorization sees the resulting JSON values, not unvalidated input.
Required fields stay required; Optional does not imply a default. Output annotations
are validated by the built-in executor and direct tool execution. MCP output schemas
validate successful structured content; behavioral annotations never grant permission.

`Tool.prepare(ToolCall(...))` captures a frozen Invocation. Defaults merge in this order:
Python signature defaults, missing top-level JSON Schema defaults, configured
`default_params`, then caller arguments. Callers may override configured defaults;
use policy to prohibit an override. Validate the signature/schema before policy.
Nested policy arguments are immutable; observers receive detached copies. Typed
function arguments and callable identity are privately captured for execution.
This is an API integrity boundary, not a sandbox against hostile Python introspection.

Inject an `Executor` with `execute(invocation)` and `async aexecute(invocation)` into
Model or Agent. The executor owns effects, cancellation and output validation if it
replaces LocalExecutor. `Model.propose`/`apropose` force autorun off; use
`model.astream(..., autorun=False)` for streaming proposals. No partial argument delta
is executable. Native IDs are retained, otherwise unique local IDs are generated.
Supply exactly one external result per ID with `model.supply_results(response, results)`.
A mapping keyed by call ID or a list of records with `call_id` may arrive out of order;
legacy ordered lists remain supported. Incomplete tool exchanges cannot be sent as
normal provider history until resolved. Neither MCP nor execution automatically
replays an uncertain tool effect.

Agent defaults to permissive execution of its registered tools. Set
`policy_required=True` and provide `permission(name, effective_arguments)` to require
a decision. Async permission is supported only on async paths. Denial returns a
correlated error result without an effect. Observer failures do not become policy.

## Async scheduling, outcomes and cleanup

LocalExecutor, Flow, Pipeline and EventBus run synchronous functions inline by default
to preserve thread affinity. Supply a `ThreadScheduler(workers=N)` explicitly to
allow blocking work in a bounded pool. Calls remain ordered; no implicit parallel
execution of tool effects is introduced. Capacity stays occupied until the actual
worker finishes, even if its awaiting task is cancelled. Close owned schedulers;
Python cannot forcibly stop an already-running thread. Provider media conversion
performs bounded reads (16 MiB, URL timeout 10 seconds); prepare/cache media outside
latency-sensitive loops. Together's synchronous stream uses a bounded worker and
queue; other adapters use native async streaming APIs.

Session outcomes are `completed`, `failed`, `cancelled`, or `exhausted`. The compatible
successful state is `done`; unsuccessful states match their outcomes. `reset()`
returns a non-running session to idle while retaining history. Streaming interruption
retains accumulated Agent text/proposals with `extensions.incomplete=True`; Model does
so when automem is enabled. Effects start after generation finishes. Cancellation
propagates and owned stream iterators close. Consumers must close an abandoned async
iterator (`await stream.aclose()` or `contextlib.aclosing`). Cancellation during an
effect does not establish whether it happened. Resolve incomplete exchanges explicitly.

Registry and event subscriptions return handles with `.close()`. Registry callbacks
run after atomic publication; exceptions are reported without rollback, and reentrant
mutation is rejected. EventBus defaults to warning/reporting errors, offers strict
`errors="raise"`, and awaits async observers sequentially for backpressure. Strict
observer errors can interrupt after an effect; they do not roll it back. Diagnostic
sinks are best-effort and never a durable action journal. `Model.close()` disposes
subscriptions and sync clients; use `await model.aclose()` for async clients too.
Cleanup attempts all clients, reports grouped failures, and permits failed closes to retry.

## MCP lifecycle and limits

`load_server` prepares and validates the complete catalog, then atomically publishes
registrations and loaded names. A failed preparation retains the previous catalog.
A load in progress cannot republish after unload/close. `load_all` uses a finite
worker pool and returns each server's names or exception, retaining independent successes.
Defaults: 4 workers, 16 connections, 128 configured servers, 30-second operations,
300-second idle timeout, 1-second reconnect backoff. Idle eviction runs on the next
operation; capacity pressure evicts the least recently used idle connection. Active
connections are not evicted. Credentials/transports remain per server/session; no
cross-credential cache is shared. These are resource bounds, not throughput claims.

- `unload_server`: remove visibility and exact registrations; retain connection and
  already captured wrappers. This is not permission revocation.
- `disconnect_server`: reject new work while draining, then close transport; retain
  catalog so a later call may establish a connection.
- `remove_server`: drain/disconnect, remove configuration and catalog, invalidate wrappers.
- `close`: stop new work, remove owned discovery/routing tools, attempt every session
  close; return grouped failures. A drain deadline still leads to a close attempt.

Calls capture the dispatch attempt, SDK session and owner task. Only a matching
failed dispatch may invalidate the current connection. Late success/failure/timeout
cannot destroy a replacement owner. A subsequent new call may reconnect after backoff;
a failed call itself is never retried. SDK contexts still enter/exit in their owner task.
Catalog limits apply after SDK decoding; they are not wire-frame memory limits.

## History and provider metadata

`ContentBlock` represents text/image/audio/resource references; bytes/persistence are
consumer-owned. Provider converters preserve supported image/audio references and
native tool IDs. Unsupported media fails through capability checks. Resource references
are portable in history but are not submitted to these provider adapters.
`History.to_dict()`/`from_dict()` use version 1. JSON-compatible results/extensions are
required for JSON encoding. A supplied token counter plus token_budget supplements the
count window; a complete call/results memory is removed as one unit. `on_truncate`
receives lost entries. Counters/policies must be reattached after deserialization.
Legacy `dedup_tools=True` removes old matching results; it is opt-in, lossy compaction,
not proof that identical arguments observed identical external state.

Finish reasons normalize to `stop`, `length`, `tool_calls`, `content_filter`, or `unknown`.
A differing native reason remains in `extensions.native_finish_reason`. SDK response IDs
are preserved where exposed; unknown usage is None and measured zero remains zero.
Native SDK exceptions retain their types/context. Retry classifies transport/timeouts,
HTTP 408/429/5xx and recognized SDK transient errors, without classifying authentication,
argument validation or programming errors as transient.

## Incremental execution and cancellation (Control readiness follow-up)

Every Model autorun and Agent batch publishes receipts to `ResponseMem.tool_results`
**by call ID after each successful call, before observers or the next call**. If a
later dispatch, output validation, observer, or cancellation fails, previous receipts
remain available. Model also exposes the response as `last_response`, including when
`automem=False`; an Agent always retains its partial response in Session history.
Model autorun now raises execution failures and stops the batch instead of returning
an error and proceeding to further effects. Unknown/blacklisted Model tools and policy
denials still produce correlated error records without starting an effect.

`response.tool_outcomes` returns a detached snapshot of
`extensions.execution = {"version": 1, "calls": {call_id: outcome}}`. Each outcome
contains the name and status:

| Status | Evidence |
| --- | --- |
| `not_started` | Synaptic has not admitted this call's effect |
| `started` | Execution has been admitted; completion is not yet known |
| `succeeded` | A validated result was received and saved |
| `denied` | A correlated rejection was saved without executing the tool |
| `uncertain` | Execution had started when control was lost; effects may have happened |
| `reconciled` | The consumer supplied an authoritative result |

History format remains version 1; extensions preserve partial outcomes and receipts
through JSON serialization. Results must be JSON-compatible. A snapshot of `started`
work is not evidence of completion. Interrupted batches retain `extensions.incomplete`
and the exception class in `extensions.execution_error`. Provider continuation rejects
incomplete responses, even when every call has a receipt but a final observer failed.
`model.supply_results`/`complete_tool_call` explicitly acknowledge the entire exchange;
include existing receipts unchanged and resolve every remaining ID. Conflicting
replacement of a known receipt is rejected, including positional-list replacement.
Execution helpers reject previously recorded/admitted calls; no recovery path replays
completed or uncertain effects. Reconciliation is an application decision, not a retry.

`abort` is accepted by Model invoke/ainvoke/astream and Agent run/arun/astream.
Synaptic checks it before generation, after generation, after awaited permission and
observation, between calls, and at the local worker's effect boundary after scheduler
admission. Task cancellation propagates. A token cannot preempt a blocking function;
the next boundary observes it and preserves any receipt already received. A worker
that has already started may outlive its cancelled awaiter and stays `uncertain` until
consumer reconciliation; late completion is not invented as a confirmed receipt.

External executors are conservatively admitted when handed an Invocation. They own
remote effects, idempotency, output validation and cancellation after that handoff.
A cooperative executor may declare `manages_admission=True`, call
`invocation.check_cancelled()` after its queue/permission waits, then call
`invocation.begin_effect()` immediately before remote submission (or
`invocation.execute()` for the captured local callable, which gates and begins itself).
Do not call both begin_effect and execute for the same effect. The gate rejects a
second begin. This permits truthful `not_started` reporting for cooperative queues;
opaque executor cancellation remains `uncertain`. Synaptic cannot stop external code
that ignores the gate or prove that a remote request was not accepted. No durable
exactly-once or process-crash recovery guarantee is made by this in-memory journal.
