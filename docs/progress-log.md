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
