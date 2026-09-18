# Handoff / next steps

T-011 addresses Control's integration review against `3eb9956`. Read
[integration-followup](integration-followup.md), [contracts](contracts.md),
[migration](migration.md) and [qualification](qualification.md). S-009 preserves this
session's before/after evidence; prior fixes and historical logs remain intact.

Fresh frozen validation: **360 tests**, **92 runtime files type-clean**, fatal Ruff,
**ten isolated wheel installs**, py.typed in wheel/sdist, and strict installed-consumer
checks in core/OpenAI/all targets (valid examples accepted, all 12 invalid lines rejected).
The offline example also passed in those three isolated installs.

```sh
uv sync --frozen --all-extras --python 3.13.8
uv run --frozen --all-extras python scripts/check_quality.py
uv run --frozen --all-extras python scripts/check_release.py
```

Immediate maintainer decision: designate and verify the private vulnerability-reporting
route, then update SECURITY.md and releasing.md. No reporting contact/channel is currently
claimed. The tracked qualification workflow is configured but has not run remotely;
only the maintainer publishes the branch, confirms that run and makes release decisions.

Retry owns only invisible effect-free attempts. Recovery receipts and unresolved records
survive context trimming/serialization; archives are consumer-retained and not bounded by
the context budget. Reconcile complete exchanges without replacing known receipts. New
history v1 fields must not be round-tripped through older readers that drop them.

Measured support stays Linux x86_64 / CPython 3.13.8, MCP 1.30.0 / protocol 2025-11-25,
and pinned SDK fixtures. No Windows/macOS, alternate-runtime/SDK, live service or scale
qualification is implied. Blocking effects can outlive cancellation; durability and grants
remain consumer-owned. Do not publish, rewrite history, read credentials or edit siblings.
