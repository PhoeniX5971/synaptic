# Finding-by-finding completion report

Scope: [original brief](hardening-brief.md) and [follow-up](followup-brief.md).
Baseline 6a8b987 passed 176 tests. The S-007 locked-environment suite: **238 passed**,
including real local MCP stdio/HTTP transports. No original finding is deferred.
Qualification boundaries below are limits of measured evidence, not planned substitutes
for implementation. See [qualification](qualification.md) for commands and versions.

| Finding | Implemented behavior | Evidence |
| --- | --- | --- |
| 1 | Lossless versioned MCP JSON envelope, typed blocks, structured presence, error/meta/provenance; explicit lossy helpers | test_mcp_results, test_mcp_errors, real mixed stdio fixture |
| 2 | Bounded paginated discovery; repeated cursor and schema checks; atomic cache | test_mcp_discovery, test_hardening_mcp |
| 3 | Attempt-owned lifecycle, coalesced readiness, startup/cancel/death/reconnect/close handling | test_mcp_lifecycle, test_mcp_dispatch, real transports |
| 4 | Shared completion assembly, optional usage, final-only/stream parity | test_completion_contract, test_provider_sdk_contract, test_provider_parity |
| 5 | First-yield and tool-effect replay barriers | test_hardening_core, test_retry_boundaries, test_retry_contract |
| 6 | General awaitables; sync rejection and coroutine cleanup | test_awaitable_contract, test_hardening_core |
| 7 | Unique bindings, explicit replacement/unbind, live registry views | test_binding_contract |
| 8 | Fixed-worker load_all, isolated failures, capacity/idle/backoff/deadlines, drain/disconnect/remove/close | test_hub_contract, test_final_boundaries |
| 9 | Exact hub ownership, canonical identity, bounded aliases, preserved schemas/annotations, declared-output validation and turn snapshots | test_followup, test_hub_contract, test_final_boundaries |
| 10 | Scoped defaults, protected snapshots, disposable callbacks/client cleanup, concurrent-owner rejection | test_registry_contract, test_final_boundaries, test_runtime_boundaries |
| 11 | Pydantic rich schemas, signature/default/input validation and output adapters | test_invocation_contract, test_tool |
| 12 | Immutable effective invocation, injectable Executor, proposal-only API and external result correlation | test_followup, test_invocation_contract, test_runtime_boundaries, test_history_contract |
| 13 | Explicit bounded ThreadScheduler for tools/flows/events, cancellation retains worker capacity, bounded SDK bridge | test_runtime_boundaries, test_provider_parity |
| 14 | Explicit capabilities, token-limit forwarding, native IDs, normalized finish/usage metadata, native async/provider-specific fixtures | test_provider_sdk_contract, test_provider_parity, provider-matrix.md |
| 15 | Typed media, versioned serialization, correlated complete exchanges, token/count truncation, opt-in lossy compaction | test_history_contract, test_adapters, test_memory |
| 16 | Distinct completion/failure/cancel/budget outcomes, partial stream records, iterator cleanup and independent observer policy | test_runtime_boundaries, test_final_boundaries |
| 17 | Transient status/SDK classification, retry-after, bounded exponential jitter/deadlines, effect barriers | test_retry_contract, test_retry_boundaries, test_run |
| 18 | Pre-sink redaction, metadata defaults, context-scoped correlation, bounded queue/event size, visible failures and rotation | test_diagnostic_contract |
| 19 | Lazy SDK extras, Python floor, uv.lock, license metadata, isolated install matrix and exact MCP support | scripts/check_installs.py, uv.lock, test_mcp_transports, qualification.md |

## Four follow-up failures

The six initial regression cases failed before the fixes (S-004).

1. Late request ownership: a call captures attempt/session/task; old success/failure/timeout
   cannot invalidate B. Current-owner failure still invalidates it. Concurrent replacement
   calls and owner cleanup are verified in test_mcp_dispatch.
2. Prefix deletion: exact owner transactions preserve overlapping names, reject cross-hub
   collisions, and independently remove discovery/routing registrations. Aliases reverse
   through canonical metadata. test_followup and test_hub_contract verify these boundaries.
3. Refresh/cleanup: preparation precedes mutation; loaded and registry publication are one
   synchronous transaction. Observer failure does not roll back state or prevent cleanup;
   every owned close is attempted. In-flight refresh cannot resurrect an unloaded/closed
   catalog. test_followup, test_hub_contract and test_final_boundaries cover these cases.
4. Policy argument mismatch: signature/schema/default validation produces one immutable
   Invocation before authorization. Effects use that captured invocation. Forbidden defaults
   produce no effect; valid defaults and caller overrides are checked consistently in sync,
   async, streaming and external-executor fixtures. Observer mutation cannot change them.

## Boundaries retained intentionally

- SDK/adapter fixtures are not live-provider qualification. Capabilities vary by model;
  exact measured SDKs and differing media/structured contracts are listed separately.
- MCP 1.30.0 / protocol 2025-11-25 is measured. Other versions/extensions are not inferred
  from dependency ranges. Catalog bounds apply after SDK decoding, not before wire buffering.
- Python 3.13.8 Linux is runtime-qualified; >=3.11 declares the language floor, not a test
  result for every interpreter/platform. T-010 supersedes the earlier typing limit: all
  92 runtime-package files now pass the full-body gate; only Vertex/xAI SDK import typing
  remains unqualified. The latest suite/install evidence is in qualification.md.
- Thread cancellation cannot undo effects; sync retry deadlines cannot preempt blocking code.
  Resource references are preserved in history but unsupported for provider submission.
- Diagnostics are best-effort. Persistence, desktop automation/grants, consumer plugin
  packaging and PC resource scheduling remain outside the framework as requested.

The subsequent Control review is resolved in [readiness-followup.md](readiness-followup.md).
The latest integration full suite passes 360 tests; original F IDs and their evidence are preserved.
