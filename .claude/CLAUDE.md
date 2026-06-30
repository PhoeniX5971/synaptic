# Engineering Standard

This project runs under a fixed working agreement. Follow it on every task.

## Core principles

1. **Small, focused files.** Target **≤ 200 lines** per source file. If a file
   grows past that, split it by responsibility into cohesive modules. This is
   enforced — a hook blocks oversized code files, so design for it up front
   rather than refactoring after the block.

2. **Simplest correct solution wins.** Prefer the lower-complexity option in
   time, memory, and structure. Reach for a clever or layered design only when a
   simpler one genuinely cannot meet the requirement — and say why.

3. **Performance by default.** Lazy-load what isn't needed yet. Load/compute
   once and reuse. Never repeat work on a hot path. Call out any change that
   adds latency or allocation to a frequently-run path.

4. **Ask, don't assume.** If intent, scope, or expected behavior is unclear,
   ask before writing code. A wrong assumption costs more than a question.
   Surface unknowns explicitly instead of guessing.

## Memory protocol — read before you build

`memory/` is the project's long-term brain. The full transcript of everything
done lives in `logs/` (auto-written); `memory/` is the curated *why*.

**At the start of any non-trivial task, read:**
- `memory/intent.md` — what the user actually wants and current priorities.
- `memory/bugs.md` — past bugs and their root causes. **Reason over this so a
  fixed bug never returns.** If your change touches a area listed here, account
  for it.
- `memory/file-map.md` — what each significant file does and why it exists.

**After a meaningful change, update memory before you consider the task done:**
- `memory/changelog.md` — append what changed and *why* (newest at top).
- `memory/file-map.md` — keep the file→purpose map current when files are
  added, removed, or repurposed.
- `memory/bugs.md` — when you find/fix a bug, record: what it was, where, the
  **root cause** (why it existed), and how it's prevented now.
- `memory/intent.md` — when goals or constraints shift.
- `memory/backlog.md` — track outstanding tasks (mirrors ClickUp once wired).

Keep memory entries short and factual. Link related entries inline.

## Git commits

Always commit as the user (PhoeniX5971) as sole author. Never add a
`Co-Authored-By` line. Never pass `--no-verify`. Use a HEREDOC for the
commit message to preserve formatting.

## Logs

`logs/session-YYYY-MM-DD.md` is written automatically by hooks — every prompt,
every response, every tool action. Do not edit logs by hand; they are the audit
trail for deep reviews. If a hook ever misbehaves, see `.claude/hooks/`.
