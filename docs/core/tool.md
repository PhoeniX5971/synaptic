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

Synaptic infers simple parameter types from annotations:

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

@registry.autotool("Get the current tenant id")
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
