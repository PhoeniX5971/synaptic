# Changelog — what changed and why

> Append-only. Newest at top. Each entry: what changed, why, and which files.
> This is the curated narrative; `logs/` holds the full raw trail.

## Format

```
## YYYY-MM-DD — <short title>
**What:** <the change>
**Why:** <the reason / problem it solves>
**Files:** <paths touched>
```

---

## 2026-06-30 — Production hardening audit (13 bugs fixed)
**What:** Full pre-public audit. Fixed 13 bugs across 10 source files in a single commit (`1f9a963`).
**Why:** Preparing Synaptic for public release; needed a clean, correct baseline.
**Files:** `synaptic/mcp/session.py`, `synaptic/signal/dsl.py`, `synaptic/signal/collector.py`, `synaptic/agent/agent.py`, `synaptic/agent/session.py`, `synaptic/core/base/memory.py`, `synaptic/core/tool.py`, `synaptic/core/model.py`, `synaptic/core/_factory.py`, `synaptic/providers/deepseek/model.py`, `synaptic/providers/gemini/model.py`, `synaptic/providers/xai/model.py`, `.claude/CLAUDE.md`, `.gitignore`

See `memory/bugs.md` for each bug's root cause.

## 2026-06-30 — MCP stdio real lifecycle fix (asyncio.Task ownership)
**What:** Rewrote `MCPSession` to own a dedicated `asyncio.Task` per session. The same task that enters `stdio_client` is the one that exits it. `close()` cancels the task; `create_future()` blocks it until cancelled.
**Why:** AnyIO cancel scopes must be exited in the same task that entered them. Prior approach called `close()` from a different task, triggering `RuntimeError: cancel scope mismatch`.
**Files:** `synaptic/mcp/session.py`

## 2026-06-30 — DSL <final> and <literal> blocks; agent termination fix
**What:** Added `<final></final>` sentinel block (signals agent run complete) and `<literal>` escape block. Added `final_seen` tracking through collector → `ResponseChunk.final` → `TurnComplete.final`. Added protocol retry loop in `astream()` with `max_protocol_retries` param.
**Why:** Agent was terminating prematurely in signal/text mode when the model emitted no `<tool>` block. No mechanism existed to distinguish "done" from "forgot to call a tool".
**Files:** `synaptic/signal/dsl.py`, `synaptic/signal/collector.py`, `synaptic/agent/agent.py`

## 2026-06-18 — Project scaffolding
**What:** Added `.claude/` standard (rules, logging + size-guard hooks), `memory/`, and `logs/`.
**Why:** Establish consistent structure, durable memory, and a full audit trail across runs.
**Files:** `.claude/CLAUDE.md`, `.claude/settings.json`, `.claude/hooks/*`, `memory/*`
