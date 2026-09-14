# Architecture

`Model` selects an adapter through `core/_factory.py`. `core/_runner.py` delegates
sync calls or aggregates async provider chunks, optionally writing `History` and
executing one batch of tools. Adapters own SDK clients, schemas and conversion.
`ToolRegistry` merges explicit bindings with scoped or legacy process-global tools.

`Agent` owns a turn loop, session history, tool permissions, events and tool
execution. It disables model autorun for invoke paths. Signal collection optionally
parses a text DSL into tool proposals. Consumer-owned execution remains planned.

`MCPHub` owns named `MCPSession` objects and exposes discovered tools in a registry.
Each session owns a dedicated asyncio task so SDK/AnyIO contexts enter and exit in
the same task. Discovery/results/lifecycle are being hardened in M-002.

`Flow` and `Pipeline` sequence callables. `Retry` wraps calls and streams.
`synaptic/logging` is the preserved opt-in diagnostic bus and JSONL sink; it is
global and currently logs payloads without privacy guarantees (finding 18).

History is in-memory and text-oriented. Persistence, desktop policy and grants
belong to consumers. Current models/agents mutate shared state; do not concurrently
share instances until isolation contracts are qualified.

Completion assembly is now shared by Model async paths and Agent streaming.
Unknown usage is explicit. Model binding changes rebuild provider schemas while
registry-derived tools remain owned by the registry. Retry observes a strict
first-yield boundary and requires safe invocation before retrying generation.
