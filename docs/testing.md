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
