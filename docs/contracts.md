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

## Integration ownership: attempts, context and recovery (T-011)

`Retry(Model(...))` owns each eligible generation attempt and holds the Model history's
operation guard through backoff. With autorun disabled, failures before any provider
output are archived in `history.failed_attempts`, not normal conversation context.
Each record retains incomplete/error metadata and `extensions.attempt_input` (prompt,
role and media references). This archive also exists with automem disabled; automem
controls conversation writes, not retry diagnostics. Repeated failures, cancellation
and deadlines preserve these records. A successful retry adds only one normal exchange.

A first provider chunk, even when collected internally by ainvoke, closes the replay
window. Synchronous generation returning a response also closes it. Visible/partial
output is not repeated; autorun effects remain non-retryable. Existing incomplete
execution is still rejected before generation, regardless of retry policy. Retry never
clears unrelated history or previous receipts. Generic retry targets retain their
caller-responsible contract and do not acquire Model's attempt archive semantics.

History separates three collections:

- `entries`: the count/token-bounded context projection submitted to providers.
- `recovery`: independently retained tool exchanges and incomplete responses, including
  entries dropped from context. Current-context exchanges share the same object here.
- `failed_attempts`: invisible, effect-free failed Retry generation records.

Framework receipt publication and external reconciliation enforce attached History
budgets immediately. `effective_mems()` and Model's pre-provider check enforce them
again, including after later object growth. Truncation removes whole entries (a call
and its results remain one exchange). A single oversized exchange is omitted entirely
from provider context with the existing truncation callback/warning. Its receipts remain
in recovery. Unresolved recovery blocks continuation even when omitted from context;
reconcile that response explicitly before proceeding. The token budget bounds only
history under the supplied estimator, not instructions, the current prompt, provider
framing, or total process memory. It is not a claim of measured provider token usage.

Recovery and failed-attempt records are not automatically pruned by context budgets.
Consumers own their persistence, retention/access policy and eventual disposal; deleting
these records deliberately abandons that recovery evidence. Histories may contain
sensitive prompt/tool data. Detached observer copies cannot enforce the original
history's budget. New readers accept old history v1 records and add optional `recovery`
and `failed_attempts` sections. Do not round-trip these records through older releases
that do not preserve those sections. Counters/budget policies must still be reattached
after restoration; archive objects and active entries are restored without stale aliases.

Reconciliation validates all identities before changing receipts or outcomes. Mapping
keys must match any embedded call_id; tagged lists must be entirely tagged, unique and
match the complete proposed ID set. Mixed tagged/positional lists, missing/unexpected IDs,
invalid IDs and contradictory mappings fail atomically. Untagged complete positional
lists remain supported. Recorded partial results require explicit IDs; ambiguous legacy
partial lists and malformed restored identities are rejected. Canonical identity follows
the proposal set, so arbitrary embedded payload IDs cannot bypass receipt immutability.

## Tool timing and provider JSON text (T-012)

- `tool_end` observers receive a detached receipt copy with integer `latency_ms`. It
  measures the executor call only, on a monotonic clock; preparation, `tool_start`
  observers and permission policy are excluded. Denied calls report 0.
- The value is journaled as `extensions["execution"]["calls"][id]["latency_ms"]` for Agent
  and Model one-shot dispatch when an executor ran. It is optional: reconciled, legacy
  and unregistered-call records have none. `ToolCallResult.latency_ms` and the
  `agent.tool` log event carry the same number.
- Receipts in `tool_results` are unchanged and remain the only tool data sent to providers.
- A call that raises ends the batch without `tool_end`, so it has no latency record.
- JSON text that Synaptic builds for providers (Chat Completions tool arguments/results,
  xAI arguments/results/schema, Claude and xAI structured-output text) keeps non-ASCII
  characters literal. `synaptic.set_ensure_ascii(True)` restores escaping process-wide.
  The internal `dedup_tools` key and diagnostic writer are unaffected.

## Uniform receipts, native provider tool results (T-013)

Synaptic keeps one receipt shape in history: `{"name", "call_id", "result" | "error"}`.
`providers/_results.py` is the only reader; each adapter wraps the value the way its SDK
is meant to be used, and correlation travels only in the provider's own call-ID field.

| Adapter | Correlation | Success content | Error content |
| --- | --- | --- | --- |
| Chat Completions (OpenAI-compatible) | `tool_call_id` | string as-is, otherwise JSON | `{"error": message}` JSON |
| Claude | `tool_use_id` | string as-is, otherwise JSON | message with `is_error: true` |
| Gemini | `FunctionResponse.id` and name | `{"result": value}` | `{"error": message}` |
| Vertex | function name (SDK has no ID) | `{"result": value}` | `{"error": message}` |
| xAI | `tool_call_id` | string as-is, otherwise JSON | `{"error": message}` JSON |

- `name`/`call_id` are no longer repeated inside tool content. Plain string results are
  not JSON-quoted, so `set_ensure_ascii` affects only structured values and arguments.
- A dict that is not a receipt (no `result`/`error`) is sent whole.
- `ResponseMem.latency_ms` is the model-call duration set by the runner for invoke,
  ainvoke and astream, visible to `step_end` observers. It is not serialized; restored
  and hand-built responses report None.
- Synchronous agent runs emit the `ToolCallResult` signal like async runs. Signal
  handlers on that path must be synchronous. `ToolCallResult.error` carries the receipt's
  error message, or None.

## Parallel tool dispatch (T-014)

- Opt-in twice: the tool carries `metadata["parallel"]` and the Agent has
  `max_parallel_tools > 1`. Default behaviour is unchanged sequential dispatch.
- Applies to `Agent.arun`/`astream`. `Agent.run` and Model one-shot dispatch stay sequential.
- Consecutive marked calls form a batch bounded by a semaphore; unmarked, unknown or
  blacklisted calls run alone and separate batches.
- Receipts are appended in call order after the batch settles, never completion order,
  so positional and call-ID correlation both hold. `tool_end`, signals and logs follow
  in call order. `tool_start` observers and permission policy run concurrently per call.
- If a call fails, its siblings run to completion and their receipts are recorded; the
  failed call is journaled `uncertain`, the response is marked incomplete and the first
  failure in call order is raised. Cancelling the run cancels the whole batch.

## Run ledger and replay (T-015)

- `synaptic.Ledger` appends ordered entries (`seq`, `at`, `kind`, `session_id`, data) and
  never edits or removes them. Reads return deep copies. Appends are thread-safe.
- `Agent(ledger=...)` or `ledger.attach(agent)` records `run_start`, `input`, `model`,
  `tool_start`, `tool_end`, `run_end`. Lifecycle entries are written directly by the
  agent; model and tool entries arrive through the agent's EventBus, so they follow its
  observer-failure policy. One ledger per agent at a time; `attach` returns a subscription.
- `sink(entry)` is the storage hook. Sink failures warn (`RuntimeWarning`) and do not
  interrupt the run or remove the in-memory entry.
- `fork(seq)` copies a prefix. `synaptic.run.replay`/`areplay` feed each recorded run's
  first input to another agent and return a new ledger; effects are really executed and
  media inputs are not re-sent. `compare` reports both summaries and integer deltas.
- The ledger is evidence, not state: resume and reconciliation still use History.
- Streamed agent responses now carry `latency_ms` like `run`/`arun` responses.

## Task state, context compilation, resume, handoff and delegation (T-016)

- `Session.task` is a consumer-owned JSON-compatible dict, exposed as `Agent.task`.
  Synaptic never reads or writes it. `Session.to_dict()`/`from_dict()` (version 1) carry
  id, history, metadata, task and outcome; a restored session is always `idle`.
  History construction options (size aside) and compilers are not serialized.
- `History.compiler(mems) -> mems` projects the provider context after budgeting and
  deduplication, on every `effective_mems()` call. Canonical entries, recovery records
  and continuation checks are unaffected. `Agent(context=fn)` installs
  `fn(entries, task)` on the session history at each turn boundary.
- Resume is `Agent(..., session=restored).run(None)`. Unresolved exchanges still block
  continuation after restoration. No effect is replayed.
- `Agent.handoff(model)` is applied at the next turn boundary (including the first turn
  of the next run). Tools are unbound from the previous model and bound to the target;
  the target receives the session history. An active target raises at that boundary and
  fails the run. The in-flight response is not interrupted. Recorded as `agent.handoff`
  on the logging bus and `handoff` in the ledger.
- `Agent.as_tool` / `subagent` wrap an agent or factory as a Tool taking `task: str` and
  returning the final message. Factory delegates are marked parallel-safe; single-agent
  delegates are not. Delegate failures and budget exhaustion surface as tool failures.
  Parent cancellation propagates through task cancellation; a parent `abort` event is
  not passed to the delegate.
