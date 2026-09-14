# Intent — what the user wants

> The north star for this project. Keep it current. Read at the start of work.

## Project goal

Synaptic is a provider-neutral LLM orchestration framework (Python). It abstracts OpenAI, Anthropic, Gemini, DeepSeek, xAI behind a unified `Model` / `Agent` API with tool calling, streaming, MCP, and a signal/DSL mode for models that can't do native tool calls.

## Current priorities

- [ ] Public release readiness — code is clean post-hardening; next step is docs/examples  · priority: H
- [ ] README / quickstart guide  · priority: H
- [ ] PyPI packaging (`pyproject.toml`, versioning)  · priority: M

## Constraints & preferences

- Files ≤ 200 lines (hard limit enforced by hook). Simplest correct solution. Performance by default (lazy load, load once, no repeated work on hot paths). Ask before assuming.
- No Co-Authored-By in commits; always commit as PhoeniX5971 (sole author); never --no-verify.
- asyncio-only for MCP (no anyio abstractions needed beyond what MCP SDK pulls in).
- Signal/DSL mode (`<tool>`, `<message>`, `<final>`, `<literal>`) is TEXT or AUTO `SignalMode`.

## Out of scope / explicit non-goals

- Multi-agent orchestration beyond single `Agent` class.
- Built-in UI or serving layer.
