# Synaptic Docs

This folder is the practical guide to Synaptic. Read in this order if you are
new to the framework:

1. [`quickstart.md`](quickstart.md)
2. [`Overview.md`](Overview.md)
3. [`core/model.md`](core/model.md)
4. [`core/tool.md`](core/tool.md)
5. [`core/agent.md`](core/agent.md)
6. [`core/memory.md`](core/memory.md)
7. [`core/streaming-events.md`](core/streaming-events.md)
8. [`core/flows.md`](core/flows.md)
9. [`providers/index.md`](providers/index.md)

Provider-specific notes:

- [`providers/openai.md`](providers/openai.md)
- [`providers/gemini.md`](providers/gemini.md)

Runnable examples live in [`examples/`](examples/).

## Production and handoff records

Read [next steps](next-steps.md), [roadmap](roadmap.md), latest
[progress](progress-log.md), and [development plan](development-plan.md) first.
Then consult [architecture](architecture.md), [implementation rationale](implementation-rationale.md),
[decisions](decisions.md), [known issues](known-issues.md), [testing](testing.md),
[debugging](debugging.md), and [technology/compatibility](tech-stack.md).
The [hardening brief](hardening-brief.md) preserves requested acceptance criteria.

Stable IDs: M milestones, T tasks, D decisions, BUG defects, F findings, S sessions.
Never reuse IDs. Progress is append-only, oldest first. Archived plan and memory
under `archive/` preserve history and are not current release-readiness claims.
Raw local transcripts under `logs/` are ignored and are not portable project memory.

See [migration notes](migration.md) for changed public contracts.
