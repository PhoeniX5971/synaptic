# Synaptic engineering workflow

Use the production-coding-workflow and coding-agent-handoff-protocol skills.
Canonical project memory starts at `docs/README.md`: read roadmap, latest progress,
next steps, development plan, relevant architecture, rationale, decisions, issues
and testing notes before editing. Update those records with meaningful changes.
Historical `memory/` and `PLAN.MD` are archived in `docs/archive/`.

Keep source modules focused and at most 200 lines when adding or changing them;
record inherited oversized files instead of performing unrelated rewrites.
Prefer simple, explicit boundaries and actionable engineering records.
Make routine decisions within authorized scope and preserve existing work.

Commit cohesive changes with matching records. Author and committer must both be
PhoeniX5971 <moh845195@gmail.com>. Check identity before each commit; no authorship
trailers, bypassing hooks, publication, remote changes or history rewriting.
Use `type(scope): imperative summary [T-NNN]` subjects and Why, Changes, Validation
body sections. Only the user publishes.

Run deterministic tests without credentials or network. Live demos in
`examples/live/` require explicit invocation. Record commands and limitations;
do not treat mocks as transport or provider qualification. Keep consumer desktop
policy, grants and persistence outside Synaptic.

`docs/progress-log.md` is the append-only curated work log (oldest first).
`logs/` contains ignored local raw transcripts: preserve existing logs, never
commit credentials or raw transcripts, and do not claim hooks run across agents.
Local `.claude/` hooks/settings are machine-specific and do not travel with clones.
