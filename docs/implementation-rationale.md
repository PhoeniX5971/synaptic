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
