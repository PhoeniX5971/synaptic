# Testing and qualification

Run deterministic tests from the repository root:

```sh
.venv/bin/python -m pytest tests -q
```

Tests use `asyncio.run`; pytest-asyncio is not required. Provider tests stub clients
and make no real calls. Live exploratory scripts live in `examples/live/` and are
never collected. Do not read `.env` or use credentials for deterministic checks.

Baseline selected suite: 102 passed, 9 failed, 5 warnings (S-001). Failures reflect
stale expectations; new seven-defect regressions must fail before fixes.

No existing real MCP fixtures, clean-install matrix, type-check baseline or lockfile
was found. Track added commands/versions/results below as they are actually run.
Do not infer interoperability or scalability from mock tests.

## Current checks

See [qualification](qualification.md) for exact before/after results and versions.
The complete suite includes real local transports; in a restricted sandbox run
`--ignore=tests/test_mcp_transports.py` for in-process tests, and run transport tests
with local subprocess/socket permissions. RuntimeWarning is treated as an error
in the final suite command. The only remaining warnings are expected history
truncation notices. MCP tests require the MCP extra; live-provider demos remain
manual. No external provider calls are part of qualification.
