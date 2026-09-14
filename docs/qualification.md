# Qualification — first hardening tranche

## Measured compatibility

| Component | Version / scope | Evidence |
| --- | --- | --- |
| Python | CPython 3.13.8 Linux | Full deterministic suite |
| MCP SDK | 1.30.0 | Real stdio and loopback Streamable HTTP fixtures |
| MCP protocol | 2025-11-25 | Initialization asserted on real stdio fixture |
| HTTP | httpx 0.28.1, local uvicorn/Starlette | Per-session headers, discovery, structured result, cleanup |
| Core completion | Fake provider chunks / stubbed adapters | Sync/async/stream/Agent parity; no live-provider qualification |
| Result data | JSON scalars/objects/arrays/null and SDK content models | Mixed text/image, audio, embedded/resource links, annotations/meta |

MCP extra is bounded `>=1.30,<2`; only 1.30.0 has been tested. Other protocol
versions/extensions, Python versions and dependency combinations are unqualified.
Core import without provider SDKs remains unsupported; no clean-install matrix or
reviewed development lock is claimed in this tranche.

Official version-specific API references reviewed during implementation:
[ClientSession 1.30.0](https://github.com/modelcontextprotocol/python-sdk/blob/v1.30.0/src/mcp/client/session.py),
[Streamable HTTP 1.30.0](https://github.com/modelcontextprotocol/python-sdk/blob/v1.30.0/src/mcp/client/streamable_http.py).
The installed signatures were also inspected with `inspect.signature`.

## Commands and observed results

- Original selected suite: **102 passed, 9 failed, 5 warnings**; command in S-001.
- Reconciled baseline `.venv/bin/python -m pytest tests -q`: **111 passed, 4 warnings**.
- Before implementation `.venv/bin/python -m pytest tests/test_hardening_core.py tests/test_hardening_mcp.py -q`: **7 failed**.
- Core fix suite excluding MCP regressions: **137 passed, 2 expected history warnings**.
- `.venv/bin/python -m pytest tests --ignore=tests/test_mcp_transports.py -q`: **172 passed, 2 expected history warnings** before the final reconnect race test.
- `.venv/bin/python -m pytest tests -q -W error::RuntimeWarning`: **176 passed, 2 expected history warnings**, including the final reconnect race test.
- `.venv/bin/ruff check --target-version py311 --select E9,F63,F7,F82 synaptic tests`: passed.
- `.venv/bin/mypy --follow-imports=skip --check-untyped-defs synaptic/core/_awaitables.py synaptic/core/_completion.py synaptic/mcp/_connection.py synaptic/mcp/_discovery.py synaptic/mcp/_transport.py synaptic/mcp/results.py synaptic/mcp/errors.py synaptic/mcp/session.py`: passed (8 focused modules, imported implementations skipped).
- Ruff formatting (line length 100) applies to new helpers/MCP implementation/regressions. Existing repository-wide formatting/type debt was not rewritten.
- `git diff --check`: passed.

Real transports need local subprocess/socket permissions. The managed sandbox
blocked loopback socket creation and timed out stdio startup. The exact transport
tests passed after escalation; they contact no external services or credentials.
HTTP fixture initially used bare `dict`, which the SDK did not infer as structured
output; changing the fixture annotation to `dict[str, str]` qualified that path.

## Limits

Page, catalog count, decoded byte size and elapsed discovery time are tested.
Byte rejection happens after SDK decoding; it is not a wire-frame memory cap.
Idle death detection takes at most configured ping interval plus request timeout,
subject to event-loop scheduling. Sync functions still execute inline and can delay
the loop. Shutdown is bounded at the caller; a cancellation-resistant owner remains
quarantined until it exits. No throughput or unbounded-scale claim is made.

After extracting the shared short-deadline test factory to keep modules focused,
`.venv/bin/python -m pytest tests/test_mcp_lifecycle.py -q` passed all 7 tests.
