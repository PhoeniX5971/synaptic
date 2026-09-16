# Testing and qualification

Use the reproducible environment:

```sh
uv sync --frozen --all-extras
uv run --frozen pytest tests -q -W error::RuntimeWarning
```

The suite passed 238 tests on Python 3.13.8 Linux. Tests use asyncio.run; no
pytest-asyncio plugin is required. Provider fixtures use installed SDK types or
httpx.MockTransport, not live credentials. Manual demos under examples/live are never
collected. Do not read .env or contact providers for deterministic checks.

Real MCP stdio/subprocess and loopback HTTP tests require ordinary local permissions.
The managed sandbox may also prevent cross-thread event-loop wakeups used by scheduler
and mock-HTTP fixtures. A sandbox timeout is not by itself an implementation failure;
rerun the same local fixture with approved permissions. Do not skip these cases in
release qualification. The real transports perform no external service requests.

Follow-up regressions cover dispatch generations, atomic ownership/publication and
immutable effective arguments. Further suites cover bounded workers/draining, turn
snapshots, registry isolation, scheduler cancellation, Session outcomes, SDK parity,
media/history correlation, retry timing and diagnostic privacy/size/delivery.

See qualification.md for exact commands, dependency versions, final wheel install matrix,
warning explanations and static-check limits. `scripts/check_installs.py` creates only
isolated temporary environments. Tests need all extras; a core installation is checked
separately rather than expected to run provider fixture modules without their SDKs.
