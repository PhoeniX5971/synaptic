# File map — what each significant file does and why

> Keep current as files are added, removed, or repurposed. Skip the obvious;
> record purpose and rationale, not a directory dump.

## Format

```
- `path/to/file` — <what it does> · <why it exists / key decisions>
```

---

## .claude (the standard, not app code)

- `.claude/CLAUDE.md` — the working agreement (rules + memory protocol), loaded every session.
- `.claude/settings.json` — wires the logging and size-guard hooks.
- `.claude/hooks/log.py` — writes the full human-readable audit log to `logs/`.
- `.claude/hooks/guard_filesize.py` — hard-blocks oversized source files.

## Application files

### Core
- `synaptic/core/base/memory.py` — `Memory`, `UserMem`, `ResponseMem`, `History`, `complete_tool_call` · sliding-window history, dedup_tools, tool result attachment
- `synaptic/core/tool.py` — `Tool`, `ToolCall`, `ToolRegistry`, `autotool`, `collect_tools` · schema inference, registry pub/sub, method binding via `__get__`
- `synaptic/core/model.py` — `Model` base class · `bind_tools` (dedup), `history` setter, provider-neutral interface
- `synaptic/core/_factory.py` — `create_model()` factory · maps provider string → adapter; raises `ValueError` on unknown provider
- `synaptic/core/_runner.py` — sync/async tool execution helpers

### Providers
- `synaptic/providers/gemini/model.py` — Gemini adapter · system prompt via `GenerateContentConfig.system_instruction`; stream + invoke
- `synaptic/providers/deepseek/model.py` — DeepSeek (OpenAI-compat) adapter · `synaptic_tools` set before early return in `_convert_tools`
- `synaptic/providers/xai/model.py` — xAI (Grok) adapter · null guard on empty stream
- `synaptic/providers/openai/model.py` — OpenAI adapter
- `synaptic/providers/anthropic/model.py` — Anthropic/Claude adapter

### Agent
- `synaptic/agent/agent.py` — `Agent` · `run`, `arun`, `astream`; DSL text mode; `max_protocol_retries` protocol retry loop; tool execution; event bus wiring (200 lines exactly)
- `synaptic/agent/session.py` — `Session` · state machine (`idle → running → done`); `reset()` for reuse
- `synaptic/agent/events.py` — `EventBus` · sync/async event emission

### Signal / DSL
- `synaptic/signal/dsl.py` — `DSLParser` · incremental XML-like DSL state machine; `<tool>`, `<message>`, `<final>`, `<literal>` blocks; `build_dsl_instructions()`
- `synaptic/signal/collector.py` — `collect()` async generator · wraps model stream; parses DSL; propagates `final_seen` to `ResponseChunk.final` and `TurnComplete.final`
- `synaptic/signal/events.py` — signal event types (`BlockDelta`, `BlockDone`, `TurnComplete`, `ToolCallResult`, `TextDelta`)

### MCP
- `synaptic/mcp/session.py` — `MCPSession` · task-per-session lifecycle; `_run_stdio` owns enter/exit of `stdio_client`; `close()` cancels task; `_ready` event + `_error` for async init propagation
