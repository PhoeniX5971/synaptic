# Core: Tools

Tools are Python callables exposed to model providers through function calling.

## `autotool`

Use `autotool` for most tools:

```python
from synaptic import autotool

@autotool(
    description="Search an inventory",
    param_descriptions={"sku": "Product SKU"},
)
def find_product(sku: str) -> dict:
    return {"sku": sku, "stock": 12}
```

Synaptic uses Pydantic 2 for parameter schemas, including nested collections/models,
Optional/unions, Literal and enums. Unsupported annotations raise TypeError.
Primitive mappings include:

- `str` -> `string`
- `int` -> `integer`
- `float` -> `number`
- `bool` -> `boolean`

Parameters without defaults are required.

## `Tool`

Use `Tool` directly when you need a custom declaration:

```python
from synaptic import Tool

tool = Tool(
    name="add",
    declaration={
        "name": "add",
        "description": "Add numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
    function=lambda a, b: a + b,
)
```

`Tool.run(**kwargs)` merges `default_params` before runtime args.

## Async Tools

Async functions are supported by `ainvoke`, `astream`, and `Agent.arun`.
Calling an async tool through sync `invoke` raises a runtime error.

## ToolRegistry

`ToolRegistry` isolates tool scope.

```python
from synaptic import ToolRegistry

registry = ToolRegistry()

@registry.autotool(description="Get the current tenant id")
def tenant_id() -> str:
    return "tenant_a"

model = Model(..., tool_registry=registry)
```

Explicit model tools override registry tools with the same name.

## Tool Results

Tool execution returns dictionaries:

```python
{"name": "add", "result": 5}
{"name": "add", "error": "Permission denied"}
```

These are stored on `ResponseMem.tool_results`.

Tool definitions are immutable snapshots. Defaults merge and arguments validate before
Agent permission checks. Callers may override defaults; policy sees the effective values.
Model autorun result records retain the shown shape; Agent records also include call_id.
See [contracts](../contracts.md) for Executor injection and [migration](../migration.md).

## Non-ASCII arguments and results

Tool arguments, tool results and structured output are serialized for providers with
non-ASCII text kept literal (`ensure_ascii=False`), so Arabic and other non-Latin text
is not expanded into `\uXXXX` escapes. Call `synaptic.set_ensure_ascii(True)` once at
startup only if an endpoint requires ASCII-only JSON text.

## Parallel-safe tools

Tools run one at a time unless you opt in on both sides:

```python
@autotool("Fetch a URL", parallel=True)
async def fetch(url: str) -> str: ...

agent = Agent(model, tools=[fetch], max_parallel_tools=4)
await agent.arun("Compare these three pages ...")
```

Mark a tool `parallel=True` (or `Tool(..., metadata={"parallel": True})`) only when
concurrent calls cannot interfere with each other. Consecutive marked calls in one model
response run together, up to `max_parallel_tools`; an unmarked call is a barrier that runs
alone, in order. Synchronous tool functions overlap only when the executor has a
scheduler, e.g. `LocalExecutor(ThreadScheduler(...))`; otherwise they block the loop.

## Failing tools

`@autotool(..., errors="return")` sends this tool's exceptions to the model as an error
result instead of ending the run; `errors="raise"` forces the opposite. Without it the
agent's `tool_errors` setting applies. See the agent guide.
