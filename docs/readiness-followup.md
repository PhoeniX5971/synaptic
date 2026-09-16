# Control readiness review follow-up (T-010)

Reviewed Control's `docs/synaptic-readiness-review.md` against Synaptic baseline
`cac43ac2c5184926b4e1b89c924e76346e961969` on 2026-09-16. Control was read-only.
This record supplements the original F-001–019 finding report; Control's BUG-010/011
identifiers belong to that review and do not renumber earlier Synaptic findings.

| Review finding | Resolution | Regression / evidence |
| --- | --- | --- |
| Control BUG-010: a later tool failure erases earlier receipts | Publish each result before observers or later dispatch; serialize call-ID outcomes and incomplete state | `test_execution_recovery.py`, `test_dispatch_boundaries.py`, `test_execution_reconciliation.py` |
| Control BUG-011: cancellation after final yield still dispatches | Gate after generation, permission/observation, and local scheduler admission; carry token into Invocation | Same suites: all applicable sync/async/stream paths, opaque/cooperative executors, task cancellation and thread queues |
| Whole-project typing was not a reproducible gate | Check every runtime source and untyped body; fix owned state contracts, Optional values, SDK input types and inferred containers | `scripts/check_quality.py`, pinned tools/stubs, normal import following, no incremental cache; 90 source files clean |
| Support claims exceeded measured qualification | Retain exact Linux/Python/MCP/provider fixture versions and require new qualification for expansion | `qualification.md`, `known-issues.md`, README and testing instructions |

The five original acceptance cases failed before implementation: three lost receipts
and two post-final-yield cancellation violations. The expanded suite also checks strict
observer failures, output validation failure, token cancellation after a successful
call, task cancellation during a second call, generation interruption, scheduler admission,
external executor admission, history round trips, conflicting reconciliation and replay
rejection. The pre-fix full-body mypy command reported 132 errors in 28 of 88 files;
the old 81-error report used a different scope/configuration and is historical evidence.

The integration retains earlier lifecycle, ownership, retry, provider and packaging fixes.
The full deterministic gate and ten isolated wheel installs are recorded in qualification.md.
No live providers, Windows, alternative Python versions, remote CI runs, or load/scale
qualification are claimed. Running remote effects and blocking threads may outlive
cancellation; opaque executor submission remains uncertain. Persistence and authoritative
reconciliation belong to the consumer. See contracts.md for exact recovery semantics.
