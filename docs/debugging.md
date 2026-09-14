# Debugging

First verify whether an issue belongs to committed code or preserved working edits.
`git status --short`, `git diff`, and `docs/progress-log.md` identify the baseline.

For async MCP hangs, inspect state, owner-task termination and startup readiness.
Use bounded asyncio timeouts in regressions. Never enter/exit SDK transports in
different tasks. Preserve the first startup error for every concurrent waiter.

Historical tests were ignored, so “suite passed” in old memory is not evidence for
this checkout. Avoid running manual provider demos when reproducing offline bugs.
