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
