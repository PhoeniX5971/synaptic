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
