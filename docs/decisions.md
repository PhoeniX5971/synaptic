# Decisions

## D-001 — 2026-09-14: canonical production records

Use `docs/` for current state and oldest-first append-only progress. Archive the
old plan and memory verbatim rather than losing rationale or retaining conflicting
sources of truth. Raw transcripts stay local and ignored. Existing tracked Claude
instructions remain tracked; local hooks are not portable clone guarantees.

## D-002 — 2026-09-14: preserve the actual starting tree

The reviewed HEAD is `1f9a963c9fe68c6951410072eb6469c1572ce59b`. Existing logging and
permission changes are preserved as baseline work, not represented as new fixes.
The historical tests were ignored and stale. Track deterministic tests, reconcile
expectations with committed behavior, and separate manual live demos.

## D-003 — 2026-09-14: staged qualification

Start with seven reproducible defects; public APIs may change with migrations.
Keep remaining improvements explicitly planned. Fake providers can establish
framework accounting, but cannot qualify a live provider or real MCP transport.

## D-004 — 2026-09-14: explicit lossy compatibility and bounded ownership

Use a JSON-only versioned MCP envelope by default, retaining text extraction and
legacy flattening as explicit conveniences. A dataclass-only result would require
custom serialization in every existing provider tool-result path. JSON dictionaries
fit those consumers while preserving modalities and provenance.

A failed connection requires an explicit reconnect. Silent reconnect during a
failed tool request could duplicate an uncertain effect. Use SDK ping for idle
liveness instead of depending on private SDK receive-task state. Preserve owner
task context-manager lifetime, quarantine slow shutdown and expose finite limits.
Qualify the official 1.30.0 SDK; bound the extra below 2.x without claiming every
future 1.x version or every protocol extension has been tested.

## D-005 — 2026-09-16: validated execution and exact ownership

Resolve/default/validate one immutable invocation before policy and capture tool definitions
per turn. Scoped registries are the default; legacy globals require explicit compatibility.
Registry publication is atomic in the supported event-loop scope; observer failure is
separate from publication. MCP identity is canonical metadata with bounded aliases and
exact hub ownership. Unload visibility is not revocation; removal drains and invalidates.

## D-006 — 2026-09-16: bounded scheduling and explicit outcomes

Keep synchronous callables inline unless an explicit bounded scheduler is provided.
Cancellation cannot prove an effect did not occur and does not free a running thread's
capacity. Reject concurrent Model/History owners instead of promising transparent isolation.
Retain incomplete streamed history and expose failed/cancelled/exhausted Session outcomes.
Observers are independent from policy; strict observation is opt-in and can interrupt work.

## D-007 — 2026-09-16: optional SDKs and measured qualification

Core depends on Pydantic/jsonschema; SDKs belong to extras and lazy imports. uv.lock pins
development/CI resolution; library consumers retain bounded ranges. Qualify exact installed
SDKs and the official MCP 1.30.0 protocol path with local transports. SDK mock-HTTP/typed
fixtures do not claim live-model support. Do not infer protocol support from version ranges.
The existing license text is packaged without changing its grant.

## D-008 — 2026-09-16: metadata diagnostics and replay limits

DiagnosticBus defaults to metadata, redacts before every sink, uses scoped correlation,
and offers bounded queue/rotation with visible loss. Durable effect recording remains
application-owned. Retry is bounded transient generation retry before output and without
autorun effects; it is not a general automatic effect-recovery mechanism.

## D-009 — 2026-09-16: persist receipts before fallible observation

Resolve Control BUG-010/011 with per-call evidence in ResponseMem extensions. Commit each
receipt before tool_end/signals or another dispatch. Stop the batch on execution failure,
retain uncertain started work and reject automatic replay. Cancellation travels through
Invocation admission; opaque external effects are uncertain after handoff, cooperative
executors own their later admission gate. Explicit reconciliation cannot erase a receipt.

## D-010 — 2026-09-16: full-package typing and bounded support claims

Use the frozen all-extras environment, mypy 1.19.1, normal imports, untyped-body checking
and disabled incremental cache for all package files. Fix project-owned errors instead of
excluding modules or skipping owned imports. Pin types-jsonschema; narrowly permit the
untyped Vertex/xAI SDK import namespaces. Windows, other Python/MCP/provider versions and
live services require fresh executable qualification before expanding support claims.

## D-011 — 2026-09-18: separate bounded context from recovery and attempts

Archive only invisible, effect-free Retry attempts independently of conversation history;
never clear unrelated incomplete records to permit retry. Hold history ownership across
backoff and preserve output/effect barriers. Retain tool/incomplete exchanges independently
of context trimming, enforce budgets after framework mutations and before submission,
and reject unresolved recovery even after context removal. Consumers own archive retention.

## D-012 — 2026-09-18: identity validation and installed qualification

Validate reconciliation keys/embedded IDs and canonical known receipts before mutation.
Reject mixed/ambiguous formats; retain complete untagged legacy lists. Package inline
metadata and test actual installed public types, including core-only environments, with
positive and negative examples. CI runs the frozen baseline and never publishes. Remote
CI, additional platforms and a verified private security route remain separate release gates.
