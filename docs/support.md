# Support and limitations

What Synaptic has been measured on, and the limits you should plan around.

## Measured support

| Check | Environment | Result |
| --- | --- | --- |
| Lint, full-package typing, test suite | CPython 3.11, 3.12 and 3.13 on Linux x86_64 | passing |
| Wheel/sdist build, isolated installs for core and every optional extra, installed-consumer typing | CPython 3.13 on Linux x86_64 | passing |

Reproduce with:

```sh
uv sync --frozen --all-extras --python 3.13.8
uv run --frozen --all-extras python scripts/check_quality.py
uv run --frozen --all-extras python scripts/check_release.py
```

Not measured: Windows, macOS, other interpreters, SDK versions outside `uv.lock`, and
live provider services. Provider adapters are verified against the pinned SDK types
and recorded HTTP fixtures; a provider or model may support less than its adapter.
Package version ranges are not a support claim.

## Limitations

**Execution**

- Tool effects are at-most-once. Synaptic never replays a tool; an interrupted call is
  recorded as uncertain and must be reconciled before the conversation continues.
  Exactly-once behaviour depends on your tools being idempotent.
- A tool that raises ends the run. It produces no `tool_end` event and no latency record.
- A cancelled thread-scheduled tool may keep running after cancellation is reported.
- Parallel tool dispatch applies to `Agent.arun`/`astream` only. Synchronous tool
  functions overlap only when the executor has a thread scheduler.

**State and storage**

- Synaptic stores nothing. Saving a `Session`, `History` or `Ledger` is your application's job.
- `History(archive_limit=...)` bounds resolved evidence; unresolved exchanges are always kept.
- Context budgets bound the history window, not total request tokens.
- `ResponseMem.latency_ms` and history compilers are not serialized.
- `dedup_tools` is lossy compaction and does not imply stable external state.

**Agent control**

- A context compiler's output is not validated; it can produce a request a provider rejects.
- Model handoff applies at turn boundaries and cannot cross signal modes.
- A subagent must have its own `Model` instance; sharing the lead's is not detected.
- A parent's `abort` event is not passed to subagents; task cancellation is.
- Ledger replay really executes tools and does not re-send recorded images or audio.

**Providers**

- Vertex correlates tool results by function name; its SDK exposes no call ID.
- The Claude adapter requires `declaration["name"]` on tools built with `Tool(...)` directly.
- `set_ensure_ascii` is process-wide, not per model.
- Resource content references are preserved in history but rejected for provider submission.

**Other**

- MCP discovery and media size limits are checked after reading up to the limit; they
  are not a sandbox against a hostile server.
- Typing: the package is fully type-checked, but dynamic SDK payloads are not strictly typed.
