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
