# Debugging

First verify whether an issue belongs to committed code or preserved working edits.
`git status --short`, `git diff`, and `docs/progress-log.md` identify the baseline.

For async MCP hangs, inspect state, owner-task termination and startup readiness.
Use bounded asyncio timeouts in regressions. Never enter/exit SDK transports in
different tasks. Preserve the first startup error for every concurrent waiter.

Historical tests were ignored, so “suite passed” in old memory is not evidence for
this checkout. Avoid running manual provider demos when reproducing offline bugs.

MCP SDK 1.30.0 streamable_http_client accepts http_client rather than headers.
Configure headers on a session-owned httpx.AsyncClient. FastMCP structured output
fixtures need a supported return annotation (`dict[str, str]`, not bare `dict`).
A socket PermissionError or stdio handshake timeout under the sandbox can be an
environment restriction: rerun the same local-only test with authorized permissions
before changing production transport behavior.

For thread/mock-HTTP stalls, the sandbox can block event-loop cross-thread wakeups.
Rerun the bounded local fixture with ordinary permissions before changing its scheduler.
For external effects, inspect the stable call_id and incomplete history marker; never
infer absence of effects from cancellation or automatically replay a failed request.
