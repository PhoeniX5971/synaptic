# Migration notes

## First hardening tranche

- ResponseMem, ResponseChunk and TurnComplete token counts default to None when
  unavailable. Use `value or 0` only when deliberately aggregating unknown as zero.
  Memories also expose finish_reason and provider extensions.
- Model.bind_tools([tool, tool]) deduplicates. A different explicit definition
  with the same name raises; pass replace=True to update it. unbind_tools(*names)
  removes local bindings. A registry-owned tool is removed with registry.unregister;
  removing a local override can reveal a same-name registry tool. Read Model.tools
  as a snapshot; use binding/registry methods to change provider schemas.
- Retry defaults to ConnectionError/TimeoutError and never restarts after any
  yielded chunk. Model autorun must be False to retry; unknown producers require
  retry_safe=True, asserting they cannot execute effects before returning/yielding.
  No generic restart/reset protocol is offered. `@retry` remains caller-controlled
  and can repeat effects; configure its exception filter and ensure repetition is safe.
- Async execution resolves Future/custom awaitable results. Sync helpers reject
  async results and close returned coroutines; they do not cancel caller-owned
  Futures or claim a rejected awaitable has not already performed work.
- Historical History.MemoryList/memoryList became entries/memory_list before this
  tranche; the recovered tests now reflect that prior migration.
