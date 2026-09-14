# Bug ledger — so bugs never come back

> Read this before working in any area listed below. Every fixed bug gets an
> entry with its **root cause**, so future runs reason over it and don't
> reintroduce it.

## Format

```
## YYYY-MM-DD — <symptom>
**Where:** <file:line / component>
**Root cause:** <WHY the bug existed — the real underlying reason, not the symptom>
**Fix:** <what changed>
**Prevented by:** <test / guard / pattern that stops recurrence>
**Status:** fixed | watching | known-issue
```

---

## 2026-06-30 — MCP stdio cancel scope mismatch crash
**Where:** `synaptic/mcp/session.py` — `close()`
**Root cause:** AnyIO cancel scopes must be entered and exited in the same asyncio Task. `close()` was called from the caller's task, but `stdio_client`'s scope was created in the session's own coroutine, so the exit happened in the wrong task.
**Fix:** Each session now owns a dedicated `asyncio.Task`. `close()` cancels it; `create_future()` blocks it. Context managers unwind cleanly inside the task.
**Prevented by:** Task-ownership pattern — `_run_stdio` / `_run_http` are the only code that enter/exit context managers; `close()` only cancels the task.
**Status:** fixed

## 2026-06-30 — MCP session.py: cleanup outside finally block
**Where:** `synaptic/mcp/session.py` — original `close()`
**Root cause:** `self.session = None`, `self._stack = None`, `self._tools = None` were inside an `if` block, so they were skipped if the task was already None, leaving stale state.
**Fix:** Cleanup assignments moved outside the conditional — they always run.
**Status:** fixed

## 2026-06-30 — Agent premature termination in signal/text mode
**Where:** `synaptic/agent/agent.py` — `astream()`
**Root cause:** In TEXT/AUTO signal mode the model emits DSL instead of native tool calls. When no `<tool>` block appeared, agent treated the turn as complete even if the model was mid-task.
**Fix:** Added `<final></final>` sentinel; `final_seen` propagated via `ResponseChunk.final`. Without `<final>` and without tool calls, agent reprompts up to `max_protocol_retries` times.
**Prevented by:** Protocol retry guard raises `RuntimeError("Agent reached max_protocol_retries")` if the model keeps ignoring the protocol.
**Status:** fixed

## 2026-06-30 — History.entries was named MemoryList (PascalCase)
**Where:** `synaptic/core/base/memory.py`
**Root cause:** Initial naming used `MemoryList` (PascalCase on an instance attribute) which is non-idiomatic Python. `memory_list` param also wasn't snake_case.
**Fix:** Renamed to `entries` and `memory_list` respectively throughout.
**Status:** fixed

## 2026-06-30 — ResponseMem.tool_calls / tool_results typed as non-Optional with None default
**Where:** `synaptic/core/base/memory.py` — `ResponseMem.__init__`
**Root cause:** Params typed `List[ToolCall] = None` — wrong type annotation, mypy/pyright would flag it.
**Fix:** Changed to `Optional[List[ToolCall]] = None` and `Optional[List] = None`.
**Status:** fixed

## 2026-06-30 — autotool: no list/dict type support
**Where:** `synaptic/core/tool.py` — `autotool`
**Root cause:** Type mapping only handled primitives; `list`, `dict`, and generic aliases (`List[str]`, `Dict[str, Any]`) all fell through to `"string"`.
**Fix:** Added `"array"` branch (matches `list` and `__origin__ is list`) and `"object"` branch (matches `dict` and `__origin__ is dict`).
**Status:** fixed

## 2026-06-30 — Model.bind_tools could add duplicate tools
**Where:** `synaptic/core/model.py` — `bind_tools`
**Root cause:** Appended unconditionally; calling `bind_tools` twice on the same tools doubled the list.
**Fix:** Added dedup via `existing = {t.name for t in self.tools}`, then filter before append.
**Status:** fixed

## 2026-06-30 — Model: spurious window(1) call on non-automem path
**Where:** `synaptic/core/model.py`
**Root cause:** `if not self.automem: self._history.window(1)` truncated history to a single entry whenever automem was off, breaking multi-turn conversations.
**Fix:** Line removed.
**Status:** fixed

## 2026-06-30 — _factory.py: unknown provider silently returned None
**Where:** `synaptic/core/_factory.py`
**Root cause:** No `else` branch for unrecognised provider strings; caller got `None` and crashed later with a confusing AttributeError.
**Fix:** Added `else: raise ValueError(f"Unknown provider: {provider!r}")`.
**Status:** fixed

## 2026-06-30 — DeepSeek: synaptic_tools not set before early return in _convert_tools
**Where:** `synaptic/providers/deepseek/model.py` — `_convert_tools`
**Root cause:** `self.synaptic_tools` was only set after the `response_format != NONE` guard, so structured-output mode left `synaptic_tools` stale.
**Fix:** Moved `self.synaptic_tools = list(all_tools.values())` before the early return.
**Status:** fixed

## 2026-06-30 — Gemini: system prompt injected as user-role content block
**Where:** `synaptic/providers/gemini/model.py`
**Root cause:** System instructions were appended to `contents` as a `"user"` message, which confuses the model and pollutes the conversation history.
**Fix:** Passed via `GenerateContentConfig(system_instruction=...)` which is the correct Gemini API field.
**Status:** fixed

## 2026-06-30 — xAI: null-deref on empty stream (last_response is None)
**Where:** `synaptic/providers/xai/model.py` — `astream`
**Root cause:** `parse_calls(last_response)` was called unconditionally; if the stream returned no chunks `last_response` was `None`.
**Fix:** Added null guard: `for tfc in parse_calls(last_response) if last_response is not None else []`.
**Status:** fixed

## 2026-06-30 — Session.reset() missing; session not reusable after arun/run
**Where:** `synaptic/agent/session.py`
**Root cause:** State transitioned to `"done"` but no method existed to reset to `"idle"`, so a session could only be used once.
**Fix:** Added `Session.reset()` — raises if state is `"running"`, otherwise sets to `"idle"`.
**Status:** fixed
