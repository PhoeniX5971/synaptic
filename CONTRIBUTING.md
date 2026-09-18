# Contributing

Start with [project records](docs/README.md), [handoff](docs/next-steps.md) and
[runtime contracts](docs/contracts.md). Use uv 0.12.10 and CPython 3.13.8 on Linux
for the currently qualified baseline:

```sh
uv sync --frozen --all-extras --python 3.13.8
uv run --frozen --all-extras python scripts/check_quality.py
uv run --frozen --all-extras python scripts/check_release.py
```

The first command set checks every runtime module, fatal lint and deterministic tests.
The release check builds wheel/sdist, verifies typing metadata, then tests ten isolated
installs, downstream valid/invalid typing and the [offline example](examples/offline.py).
Build/install checks download dependencies; fixtures never need provider credentials.
Local subprocess, loopback and thread permissions are required. Do not run live examples
or include `.env`, credentials, raw logs or customer histories in a contribution.

For a bug, provide a minimal public-API reproduction, expected/actual behavior, exact
Python/platform/SDK versions and sanitized diagnostics. Add a regression that fails
before the fix. API changes must update contracts, migration notes and qualification;
retain prior recovery, ownership and replay guarantees. Keep changed source modules
focused and at most 200 lines. Do not hide owned typing errors with broad ignores.

Keep changes cohesive and preserve unrelated work. Update architecture/rationale when
ownership changes and append meaningful validation to the progress log. The tracked
[CI workflow](.github/workflows/qualification.yml) runs the same checks; its presence
does not establish a successful remote run or qualification on another platform.
See [security reporting](SECURITY.md) for sensitive reports and [release guidance](docs/releasing.md)
for maintainer-only versioning/publication steps. Only the maintainer publishes.
