# Changelog

## Unreleased — integration qualification

- Extension seam (`Agent(extensions=...)`, `Extension`), runtime `add_tools`/`remove_tools`/`instructions`, and `FinalAnswer` run-ending structured results.
- `Model(enforce_schema=True)` sends `json_schema` on OpenAI-compatible servers such as vLLM.
- Model diagnostic events share a `model_call_id`; `model.stream.end` includes the prompt.
- One agent loop behind `run`, `arun` and `astream`; `arun` and `Model.ainvoke` are drained streams.
- `tool_errors="return"` / `autotool(errors=...)` send tool failures to the model; new `tool_error` event.
- Guards: one agent run per `Model`; validated context compiler output; subagents receive the parent's abort.
- Quality gate verified on CPython 3.11 and 3.12 in addition to 3.13.
- `History(archive_limit=N)` bounds resolved recovery and failed-attempt evidence.
- Agent control: `task` state, `context` compiler, `Session.to_dict/from_dict`, `handoff`, `as_tool`/`subagent`.
- `Ledger`: append-only run record with storage sink, fork, replay and compare.
- Fixed: streamed agent responses now report `latency_ms`.
- Opt-in parallel tool dispatch: `autotool(parallel=True)` with `Agent(max_parallel_tools=N)`.
- `tool_end`, `ToolCallResult`, the execution journal and `agent.tool` logs report
  executor-only `latency_ms`; provider-visible receipts are unchanged.
- Tool results reach each provider in its native form from one uniform receipt: value-only
  content, JSON instead of Python repr, and native error marking where available.
- `ResponseMem.latency_ms` reports model-call duration; sync agent runs emit
  `ToolCallResult`, which now carries `error`.
- Provider-bound JSON text keeps non-ASCII characters literal by default;
  `synaptic.set_ensure_ascii(True)` restores escaping.
- Safe Model generation retries retain failed-attempt evidence independently of context.
- Context budgets are rechecked after result growth and before submission; recovery
  receipts survive truncation and serialization.
- Contradictory reconciliation IDs and ambiguous formats fail atomically.
- Wheel/sdist include typing metadata; installed consumer checks validate useful public
  types and reject invalid calls in core-only and selected optional installations.
- Tracked deterministic CI, contribution, security-reporting and release guidance added.

Existing 0.1.0 hardening work and previous review fixes remain described in
[migration notes](docs/migration.md) and [support](docs/support.md). No release publication is asserted here.
