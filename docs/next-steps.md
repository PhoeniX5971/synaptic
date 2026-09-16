# Handoff / next steps

The accepted hardening scope is implemented. Start with finding-report.md,
qualification.md, contracts.md and migration.md; the latest progress entries record
before/after evidence and local commits. Runtime is committed as `76b11f9`; the
independent retry policy is `68da55c`. The full locked-environment suite passed
238 tests, including three real MCP transports; ten independent install cases passed.

For a new checkout, run `uv sync --frozen --all-extras`, then
`uv run --frozen pytest tests -q -W error::RuntimeWarning`. Local stdio/socket/thread
wakeup fixtures need ordinary local process/network permissions. They make no provider
requests. Use scripts/check_installs.py with a freshly exported lock and built wheel
for packaging checks; qualification.md contains the exact sequence.

Future release qualification should expand typing (81 known mypy errors), Python/platform
coverage and explicitly authorized live-provider tests. Those are measured limits, not
unfinished hardening implementations. Preserve the MCP 1.30.0 / 2025-11-25 qualification
boundary until new SDK/protocol fixtures pass. Do not publish, modify remotes/history,
read credentials or modify sibling projects without a new user instruction.
