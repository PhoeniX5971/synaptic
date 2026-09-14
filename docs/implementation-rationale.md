# Implementation rationale

## Baseline ownership

`mcp/session.py` must unwind transport context managers inside their owner task.
Moving teardown into the caller recreates the historical AnyIO cancel-scope bug.
Read `archive/legacy-memory/bugs.md` for prior root causes, not current guarantees.

`core/_runner.py` holds invocation bookkeeping; provider adapters translate SDK
responses. Usage must reach memories through a shared aggregation boundary rather
than independent ad hoc counters. Current final chunks contain whole-response text,
so blindly appending their text duplicates streamed output.

`core/model.py` must distinguish explicit bindings from registry-derived views.
Copying the merged view into explicit bindings makes registry removal ineffective.

`run/retry.py` must respect the externally visible output/effect boundary. Retrying
a generator from scratch after a yield cannot undo data already observed by callers.

## Repository organization

The root image moves to `docs/assets/`; exploratory scripts move to `examples/live/`.
The broad `test*` ignore concealed regression evidence and was replaced with cache
ignores. Canonical records are indexed from `docs/README.md`. Archived notes preserve
historical work without claiming their release-readiness statements remain valid.

## Core completion, binding and execution fixes (T-003/T-004)

`core/_completion.py::Completion` aggregates snapshots/deltas and optional usage in
one place for ainvoke, streaming history and Agent streaming. Unknown usage is
None; measured zero remains zero. Completion extensions are copied into memories.
Provider sync conversions still determine native metadata availability (F-014).

`core/_awaitables.py::require_sync` closes coroutine objects before rejecting them.
Futures/custom awaitables remain caller-owned. Async execution uses isawaitable;
inline synchronous work is unchanged pending explicit offloading policy (F-013).

`Model.tools` reads the adapter's current view; bindings mutate only bound_tools,
never a merged registry view. Conflicting local names require replace=True.
unbind_tools removes explicit overrides; registry removal uses unregister.

`Retry` permits safe generation retries before the first yield only. Tool autorun
disables automatic retries; unknown producers need retry_safe=True. The generic
decorator retains caller responsibility. ConnectionError/TimeoutError are narrow
wrapper defaults; SDK-specific transient exceptions must be explicitly configured.

## MCP results and discovery (T-005)

`results.py` returns a versioned JSON-only envelope: typed content dictionaries,
structured value/presence, tool error flag, metadata, extensions and provenance.
SDK models serialize by alias; arbitrary opaque objects are rejected. Text
extraction and the old flattening operation are explicit helpers. Native block
fields/annotations survive; they remain server claims, not permission grants.
`errors.py` retains JSON-RPC code/data and separates call transport failure from
tool-reported failure. Native exception context remains chained for diagnostics.

`_discovery.py` builds a candidate under page/tool/decoded-byte/time limits, rejects
repeated cursors, duplicate names and invalid input schemas, then returns it only
on success. `MCPSession.list_tools` serializes discovery with a lock and publishes
a deep copy in one assignment. Callers get independent snapshots. This bounds
retained catalogs, not SDK wire buffering; a malicious single frame can be decoded
by the SDK before this boundary rejects it. Hub registry transactions remain T-006.

## MCP lifecycle and SDK integration (T-005)

`_connection.py` owns an attempt-specific readiness event/error so simultaneous
callers see the same startup outcome. Imports/configuration run inside the guarded
owner. A cancelled waiter leaves other waiters intact; the last departing startup
waiter cancels the abandoned owner. Close cancels once and waits for a bounded
period without recancelling transport unwind. A stuck owner stays quarantined in
closing state, preventing replacement. Concurrent close completion cannot clobber
a newer owner's state. Explicit reconnect closes before starting another attempt.

`_transport.py` alone opens stdio/HTTP SDK contexts in the owner task. HTTP headers
are scoped to its own httpx client. Initialization publishes a live session only
after success; periodic SDK ping detects dead idle transports within the configured
health interval plus request timeout. Call failures invalidate the live reference;
no effectful request is replayed. Transport exceptions from initialization retain
the original SDK error; call-level JSON-RPC and transport errors use public classes.

Defaults: connection 10s, shutdown 5s, ping interval 15s, request 30s; discovery
100 pages/10,000 tools/8 MiB decoded JSON/30s total including lock/startup. These
are configurable conservative limits, not throughput or scalability claims.
