# Progress log

Append sessions oldest first; never renumber IDs. Historical notes are preserved in
`archive/legacy-memory/`. Agent fields record operations, not Git authorship.

## S-001 — 2026-09-14 — baseline and production records

Agent: Codex. Goal: clean repository, reconcile memory and begin hardening brief.
HEAD: `1f9a963c9fe68c6951410072eb6469c1572ce59b`. Starting tree: modified PLAN.MD,
package init, Agent, runner and MCP session; untracked permission/logging modules,
Claude support files and memory. Preserved source changes without claiming them
as newly implemented fixes. Binary diff safety copy: `/tmp/synaptic-before-hardening.patch`.

Read both production/handoff skills, repository instructions, existing memory,
source and history. `.gitignore` hid all tests. Existing suite selected explicitly:
`.venv/bin/python -m pytest tests/test_adapters.py tests/test_mcp.py tests/test_memory.py tests/test_model.py tests/test_run.py tests/test_tool.py tests/test_tool_registry.py -q`
reported **102 passed, 9 failed, 5 warnings**. Failures were stale expectations;
warnings included missing pytest-asyncio config and leaked Flow/Pipeline coroutines.
Environment: Python 3.13.8, pytest 9.0.2; no MCP SDK or pytest-asyncio. `pip show`
failed because this uv environment has no pip; versions read via importlib.metadata.

Archived old plan/memory; created canonical records and exact requested brief;
separated manual live scripts; retained raw logs locally; narrowed test ignores;
updated tracked Claude workflow and local memory skill pointer. D-001–003 apply.
Next: failing regressions for BUG-001–007, then fixes and measured qualification.

After reconciliation: `.venv/bin/python -m pytest tests -q` → **111 passed, 4 warnings** (two expected history truncations, two unawaited coroutines). `git diff --check` passed.

## S-002 — 2026-09-14 — core defect regressions and fixes

Agent: Codex. Baseline checkpoint: 44fc1d8, author/committer verified.
`.venv/bin/python -m pytest tests/test_hardening_core.py tests/test_hardening_mcp.py -q`
reported **7 failed** before fixes, one per reviewed defect.
Implemented shared completion accounting, general awaitables and coroutine cleanup,
explicit binding replacement/removal, and safe stream retry boundaries.
`.venv/bin/python -m pytest tests --ignore=tests/test_hardening_mcp.py -q` →
**137 passed, 2 expected history warnings**. Remaining MCP regressions are not yet fixed.
Ruff fatal checks passed with `--target-version py311 --select E9,F63,F7,F82`;
without a target Ruff misidentified inherited ExceptionGroup usage. Formatted new
helpers/regressions with Ruff (line length 100). Mypy `--follow-imports=skip` passed
for `_awaitables.py` and `_completion.py`; no whole-project typing claim.

Package install initially failed at sandbox DNS; escalated install succeeded for
MCP 1.30.0, Ruff 0.15.6, mypy 1.19.1. Inspected SDK signatures and official tagged
source: HTTP headers belong on httpx.AsyncClient, not streamable_http_client.
Next: T-005 with real stdio/loopback qualification.

## S-003 — 2026-09-14 — MCP results, discovery and lifecycle

Agent: Codex. Starting core commit: 210c10d; author/committer verified. Implemented
versioned JSON MCP envelope and explicit text/legacy helpers, bounded/validated
pagination with atomic session snapshots, attempt-specific lifecycle ownership,
finite startup/request/shutdown/health deadlines and explicit reconnect. Preserved
SDK context ownership in one task. Updated HTTP client/header API for SDK 1.30.0.

Added real subprocess stdio and loopback HTTP fixtures plus JSON/error, discovery
limits, cancellation, owner failure, repeated close and slow-shutdown regressions.
Sandbox transport run: 28 passed, 2 failed (stdio timeout and denied socket);
escalated transport run: 2 passed, 1 failed (fixture bare-dict structured annotation).
Fixed fixture annotation, then **3 real transport tests passed**. Full suite with
RuntimeWarning as error: **175 passed, 2 expected history warnings**. A final review
identified concurrent reconnect close completion clobbering a newer owner state;
added a generation guard and dedicated regression.

Focused mypy initially found missing jsonschema stubs and two type annotations;
added concrete internal annotations and an explicit import-untyped exemption for
jsonschema, then all 8 selected modules passed with check-untyped-defs. Ruff fatal
checks and diff whitespace checks passed. Exact commands and compatibility sources
are in `qualification.md`. T-001–T-005 complete; T-006 is the next action.
Remaining findings are individually reported with prerequisites in finding-report.md;
no live provider, full packaging, wire memory bound or whole-program completion claim.

Final validation after reconnect guard: `.venv/bin/python -m pytest tests -q -W error::RuntimeWarning` → **176 passed, 2 expected history warnings**. Extracted the short-deadline test factory into `tests/mcp_fakes.py`; its lifecycle suite passed 7 tests afterward. New/modified MCP source modules remain below 200 lines.

## S-004 — 2026-09-15 — follow-up reproductions and implementation

Agent: Codex. Continued from reviewed baseline 6a8b987 (176 passing tests). Read the
production/handoff skills, repository instructions and both accepted briefs. Preserved
existing work. Copied the follow-up request to docs/followup-brief.md. Six regression
cases failed before fixes: old dispatch invalidating a replacement, overlapping hub
prefixes, failed observer publication, observer-blocked cleanup, and sync/async policy
checking raw arguments rather than defaulted effects.

Implemented dispatch ownership, transactional exact registrations, canonical aliases,
immutable validated invocations and scoped subscriptions, then expanded provider/history,
scheduling/outcome, logging and packaging contracts. Intermediate in-process suite reached
190 passing tests. No provider credentials, live services, remotes or sibling projects
were touched. Continued qualification is recorded below; this entry does not substitute
plans for completion evidence.

## S-005 — 2026-09-16 — bounded retry policy (T-008)

Agent: Codex. Added transient transport/status/SDK classification, retry-after minimums,
bounded exponential jitter and total deadlines. Kept first-yield and autorun effect
barriers, general decorator caller responsibility and original exception context.
Cancellation never proves an effect absent. Sync deadlines bound retry scheduling,
not already-running functions. Runtime rationale: docs/retry-policy.md.

`.venv/bin/pytest tests/test_retry_contract.py tests/test_retry_boundaries.py tests/test_run.py -q -W error::RuntimeWarning`
passed **41 tests**. Configuration/replay regressions remain intact. Git author/committer
verified as PhoeniX5971 <moh845195@gmail.com>. Next: commit the integrated runtime
contracts and then their locked distribution qualification/handoff.

## S-006 — 2026-09-16 — reusable runtime contracts (T-006/T-007/T-008)

Agent: Codex. Runtime changes resolve all four follow-up failures and complete the
execution/provider/lifecycle/history/diagnostic boundaries. A dispatched MCP request
can invalidate only its own attempt/session/task. Hub catalogs prepare before atomic
exact-owner publication; observers cannot roll back state or skip cleanup. Worker,
connection, idle/backoff and drain limits are explicit. In-flight refresh cannot
resurrect an unloaded or closed catalog.

Tool.prepare captures immutable validated effective arguments before policy, including
defaults, typed values and callable identity. Executors/proposal-only APIs expose an
application execution boundary with stable external result IDs. Models default to scoped
registries; callbacks dispose, client cleanup attempts every owned client, and concurrent
Model/History owners are rejected. Turn snapshots prevent refresh-time rebinding.

Provider adapters now forward token limits, preserve IDs/usage/finish metadata, expose
capabilities, close interrupted sources and use compatible SDK streaming APIs. Typed
media/versioned history and complete exchanges preserve context. Agent outcomes distinguish
failure/cancellation/budget exhaustion; partial stream history remains visibly incomplete.
ThreadScheduler bounds actual outstanding workers. Diagnostics redact before any sink,
use context-local correlation, bound queues/event size and expose sink loss/failure.

Validation: full fresh locked-environment suite reached **238 passed** with two expected
history warnings and two upstream Vertex/protobuf deprecations. Real local MCP stdio and
HTTP fixtures passed. Native SDK MockTransport/typed fixtures cover all provider adapters;
no live service qualification is claimed. After public callback/Optional annotation fixes
and restoring positional registry.autotool compatibility, focused registry/tool suites
passed **25 tests**. Ruff fatal checks and ten focused mypy modules passed; whole-project
mypy improved from 108 errors/24 files to **81 errors/14 files**, not type-clean.

All changed/new source modules are <=200 lines. Nine unrelated formatter changes were
backed up, inspected and proven AST-equivalent before being removed. Automatic approval
review rejected an earlier broad restore proposal; it did not execute. No user work was
lost. Canonical runtime semantics, architecture, rationale, decisions, provider matrix
and migrations now describe the implemented behavior. Author/committer identity verified.
Next: commit final lock/install evidence and synchronize the final handoff records.

## S-007 — 2026-09-16 — distribution qualification and final handoff (T-009)

Agent: Codex. Local retry commit: 68da55c; integrated runtime commit: 76b11f9.
Built wheel/sdist and verified the uv.lock development environment. The final runtime
suite again passed **238 tests** after the last public API compatibility adjustment;
all three real local MCP transports are included. Focused mypy (ten modules), fatal Ruff,
source module <=200-line check and whitespace checks passed. Whole-project mypy remains
**81 errors in 14 files**, documented without claiming those are all inherited.

The wheel passed ten isolated installations: core, MCP, each declared provider extra,
and all extras. Each used isolated Python outside the checkout, lazy-import assertions,
selected provider imports and uv pip check. Core has no unrelated OpenAI SDK and reports
an actionable missing-extra error. The committed runtime was rebuilt and the final matrix
repeated. Lock metadata/build downloads needed network permission after sandbox DNS
failures. A missing old /tmp constraint export caused one install attempt to fail;
regenerating it resolved the setup failure. No provider calls or credentials were used.

Updated roadmap/development status, all nineteen findings, issue/qualification limits,
technology/testing guides, examples, architecture/rationale/migrations and compact handoff.
Canonical project memory lives in docs; archived history/raw local logs are preserved.
Library ranges do not imply unmeasured protocol/provider/platform support. Measured MCP
remains SDK 1.30.0 / protocol 2025-11-25, Python 3.13.8 Linux. Future live-service/type/platform
qualification is explicit release work, not an unfinished item from the accepted scope.

All accepted hardening findings are implemented. No publishing, remote/history rewriting,
sibling-project changes or consumer desktop policy were performed. The user owns publishing.
Author and committer identity verified before each local commit.

## S-008 — 2026-09-16 — Control readiness review and recovery gate (T-010)

Agent: Codex. Reviewed Control's `docs/synaptic-readiness-review.md` read-only against
clean Synaptic `cac43ac2c5184926b4e1b89c924e76346e961969`. The accepted follow-up is mapped
in readiness-followup.md; Control BUG-010/011 are review-local identifiers, not a
renumbering of the original finding history. Production/handoff records remain canonical.

Before implementation, five acceptance regressions failed: three Agent modes lost the
first receipt when the second effect raised; Model and Agent streaming dispatched after
cancellation at the final yield. A fresh full-body mypy run found 132 errors in 28 of
88 files, broader than the historic 81-error gate. These failures were not deferred.

ExecutionState now publishes each correlated receipt before observers or subsequent
calls and serializes per-call evidence in history v1 extensions. Started interrupted
calls remain uncertain; unstarted calls remain distinguishable. Model retains last_response
with automem disabled and persists before autorun when enabled. Agent retains partial
proposals even when generation is cancelled. Cancellation travels through Invocation,
permission/observer boundaries and local scheduler admission; opaque external handoffs
remain conservative, cooperative executors expose explicit admission. Reconciliation
cannot replace known receipts, including legacy ordered/mapped results, or replay effects.

Owned typing fixes add actual mixin state contracts, annotate inferred containers and
ContextVars, narrow optional SDK fields and widen Gemini's SDK union lists. JSON Schema
stubs are pinned; only Vertex/xAI SDK import namespaces lacking metadata are exempt.
The gate checks every runtime module and untyped body with normal imports and a disabled
incremental cache; no project files are excluded. Intermediate regression runs caught
empty extension changes and a writable ModelAPI.history compatibility issue; both were
corrected before qualification. A test invocation referenced a nonexistent filename and
ran no tests; subsequent focused/full runs used actual test paths. No failed validation
was counted as evidence of success.

Final validation in a freshly created `/tmp/synaptic-readiness-env` from frozen uv.lock:

- `python scripts/check_quality.py`: fatal Ruff passed; mypy **zero errors in 90 files**;
  pytest **302 passed, four warnings** with RuntimeWarning treated as an error. The warnings
  are two expected history truncations and two upstream Vertex/protobuf deprecations.
- All original 238 tests remain, plus 64 recovery/cancellation/observer/reconciliation
  regressions across applicable sync/async/stream/external paths. Real local MCP stdio,
  Streamable HTTP and lifecycle fixtures remain included. No credentials/live providers.
- `uv build --out-dir /tmp/synaptic-readiness-dist` built wheel and sdist. Exported frozen
  constraints, then scripts/check_installs.py passed core, MCP, seven provider extras and
  all extras, each outside the checkout with `python -I` and `uv pip check`. Repeated on
  the rebuilt final wheel after the final legacy receipt guard and README update.
- `git diff --check` and changed/new source <=200-line checks passed. The existing lock
  resolution was retained apart from the added pinned typing stubs and marker ordering.

Updated contracts, migration/API docs, architecture/rationale, qualification, findings,
decisions D-009/010, known issues, roadmap, development plan and portable handoff. Support
claims remain CPython 3.13.8 Linux x86_64 with the exact SDK versions in qualification.md.
Windows/macOS, other interpreter/SDK versions, remote CI, live services and scale need new
qualification. A running thread/opaque remote effect can outlive cancellation; this is not
a durable exactly-once journal. These limits are explicit, with no accepted implementation
item hidden as future work. Next: consumer integration qualification; only the user publishes.

## S-009 — 2026-09-18 — public integration and installed release gates (T-011)

Agent: Codex. Reviewed Control's docs/synaptic-integration-review.md read-only against
clean `3eb9956d9d9441dedf977c2a6ee76696e1ab42c9`. Preserved the 302-test suite, full-package
typing and earlier execution/lifecycle fixes. Four public-API reproductions failed before
implementation: Model+Retry+automem in two modes, budget growth, contradictory IDs.

Added context-local generation-attempt ownership for Retry, independent failed-attempt
retention and ownership through backoff. Output/effect barriers remain intact and prior
unresolved recovery still prevents provider entry. History now separates provider context
from recovery receipts, enforces budgets after result growth/reconciliation and immediately
before provider entry, and serializes independent recovery/attempt evidence. Oversized
exchanges leave context as whole entries; unresolved recovery cannot be hidden by trimming.
Reconciliation validates all supported formats/identities before mutation, including
restored records. Contradictory mapping keys and mixed/ambiguous formats are rejected.

Published py.typed in wheel/sdist; preserved decorator signatures, explicit exports and
public return annotations. The downstream harness runs strict mypy outside the checkout
against isolated wheel environments, accepting valid types and rejecting each of 12 invalid
lines. Core-only targets require no SDKs or development dependencies. An initial harness
run correctly produced errors but expected absolute diagnostic paths; it was fixed to
recognize relative filenames without suppressing any diagnostics. Signature preservation
exposed two inherited ModelAPI override mismatches; their required-prompt contracts were
corrected and full typing passed. All original tests remain unchanged. Final review also added canonical ordering of
restored tagged receipts before provider projection, with its own public regression;
the complete quality/release gates were repeated against that final runtime.

Tracked pinned/read-only GitHub CI now invokes the same frozen quality and release scripts.
Added contribution, security reporting, version/deprecation/release guidance, changelog and
an installed credential-free offline example. The maintainer was asked for a private
security-reporting route; no response/route has been assumed. SECURITY.md explicitly makes
route verification a public-release prerequisite. No email/channel/SLA is invented.

Final checks in freshly created /tmp/synaptic-integration-env from frozen uv.lock:

- `python scripts/check_quality.py`: fatal Ruff passed, mypy **zero errors in 92 runtime
  files**, pytest **360 passed, four warnings** with RuntimeWarning as an error. Warnings
  remain two expected history truncations and two upstream Vertex/protobuf deprecations.
- `python scripts/check_release.py`: wheel/sdist built with py.typed; **all ten isolated
  install cases passed** including uv pip check. Core/OpenAI/all each accepted valid
  public typing, rejected all 12 invalid lines and passed the installed offline example.
- Added **58 public integration regressions** covering automem on/off, modes, retry
  failures/deadlines/cancellation/backoff ownership, partial output/effects, growing
  built-in/external results, overflow, restored recovery and reconciliation formats.
- Lock consistency, whitespace and changed runtime source <=200-line checks passed.
  Build/install downloads and local subprocess/loopback/thread fixtures used normal local
  permissions. No live providers, credentials, desktop effects or publication were used.

Support remains Linux x86_64 CPython 3.13.8 and exact SDK/protocol fixtures. CI configuration
is not a remote-run result; the Ubuntu hosted image is not claimed separately qualified.
Recovery/attempt archives are not context-budget limited and require consumer retention
policy; older readers must not round-trip the new v1 fields. Durability, grants and remote
idempotency remain application responsibilities. Updated contracts, rationale, architecture,
decisions D-011/012, migration, qualification, known issues and portable handoff. Next:
maintainer reporting-route decision and eventual remote CI/release qualification.
