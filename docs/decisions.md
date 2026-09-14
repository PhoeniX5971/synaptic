# Decisions

## D-001 — 2026-09-14: canonical production records

Use `docs/` for current state and oldest-first append-only progress. Archive the
old plan and memory verbatim rather than losing rationale or retaining conflicting
sources of truth. Raw transcripts stay local and ignored. Existing tracked Claude
instructions remain tracked; local hooks are not portable clone guarantees.

## D-002 — 2026-09-14: preserve the actual starting tree

The reviewed HEAD is `1f9a963c9fe68c6951410072eb6469c1572ce59b`. Existing logging and
permission changes are preserved as baseline work, not represented as new fixes.
The historical tests were ignored and stale. Track deterministic tests, reconcile
expectations with committed behavior, and separate manual live demos.

## D-003 — 2026-09-14: staged qualification

Start with seven reproducible defects; public APIs may change with migrations.
Keep remaining improvements explicitly planned. Fake providers can establish
framework accounting, but cannot qualify a live provider or real MCP transport.

## D-004 — 2026-09-14: explicit lossy compatibility and bounded ownership

Use a JSON-only versioned MCP envelope by default, retaining text extraction and
legacy flattening as explicit conveniences. A dataclass-only result would require
custom serialization in every existing provider tool-result path. JSON dictionaries
fit those consumers while preserving modalities and provenance.

A failed connection requires an explicit reconnect. Silent reconnect during a
failed tool request could duplicate an uncertain effect. Use SDK ping for idle
liveness instead of depending on private SDK receive-task state. Preserve owner
task context-manager lifetime, quarantine slow shutdown and expose finite limits.
Qualify the official 1.30.0 SDK; bound the extra below 2.x without claiming every
future 1.x version or every protocol extension has been tested.
