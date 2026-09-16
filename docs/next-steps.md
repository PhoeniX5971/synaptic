# Handoff / next steps

T-010 / M-006 resolves Control's readiness review against `cac43ac`. Read
[readiness-followup.md](readiness-followup.md), [contracts.md](contracts.md) and
[qualification.md](qualification.md) before changing execution or support claims.
Canonical progress is S-008; previous entries and raw local logs remain preserved.

The current gate passes **302 tests**, fatal Ruff and **all 90 runtime source files**
in mypy. The final wheel/sdist were rebuilt, and all **ten isolated install cases**
passed after the final receipt guard. Use the frozen all-extras environment:

```sh
uv sync --frozen --all-extras --python 3.13.8
uv run --frozen --all-extras python scripts/check_quality.py
```

The next release action is consumer integration validation using the documented
partial-result/reconciliation contract. Do not replay succeeded or uncertain calls;
acknowledge incomplete exchanges explicitly, retaining receipts unchanged. Local
transport/thread fixtures require normal local permissions, without live provider calls.
The gate and scripts/check_installs.py are the reproducible release checks.

Qualification remains Linux x86_64 / CPython 3.13.8, MCP 1.30.0 / protocol 2025-11-25,
and the pinned provider SDK fixtures. Windows/macOS, other Python/SDK versions, remote
CI, live providers and scale remain unqualified. Vertex/xAI SDK import types are narrowly
exempt because their pinned distributions lack typing metadata. Blocking threads and
opaque remote effects may outlive cancellation; durable reconciliation is consumer-owned.

No accepted readiness implementation remains deferred. Only the user publishes. Do not
modify remotes/history, read credentials or edit sibling projects without new instruction.
