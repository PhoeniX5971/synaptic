# Backlog — outstanding tasks

> Tracks work to be done. Will mirror ClickUp once the MCP is wired (Claude
> updates status here as progress is made). Until then, maintain it by hand.

## Format

```
- [ ] <task>  · priority: H/M/L  · clickup: <id when linked>
- [x] <done task>  · done: YYYY-MM-DD
```

---

- [ ] README / quickstart guide  · priority: H
- [ ] PyPI packaging (`pyproject.toml`, version, classifiers)  · priority: M
- [ ] Provider coverage: ensure all adapters support `ainvoke` (not just `astream`)  · priority: M
- [x] Production hardening audit (13 bugs)  · done: 2026-06-30
- [x] MCP stdio lifecycle fix (asyncio.Task ownership)  · done: 2026-06-30
- [x] DSL `<final>` + `<literal>` + protocol retry loop  · done: 2026-06-30
- [x] `.claude/` scaffolding (hooks, memory, logs)  · done: 2026-06-18
