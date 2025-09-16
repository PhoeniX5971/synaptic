# Core — Tool

**File:** `core/tool.py`

`Tool` encapsulates a callable that the LLM can request via function-calling. The
tool includes a JSON-schema-like declaration used by providers.

---

## Class: `Tool`

**Type:** class

**Constructor**

```py
Tool(
    name: str,
    declaration: dict,
    function: Callable[..., Any] = lambda: None,
    default_params: dict | None = None
)
```

**Fields**

- `name: str` — unique tool identifier.
- `declaration: dict` — schema used to convert into provider function definitions.
- `function: Callable` — the callable executed when the tool is run.
- `default_params: dict` — default args merged with runtime args.

**Methods**

- `run(**kwargs) -> Any`
  - Merges `default_params` with incoming `kwargs` and calls the underlying `function`.
  - Equivalent to `function(**{**default_params, **kwargs})`.

**Error handling**

- The constructor validates `function` is callable and raises `ValueError` otherwise.
- `_run_tools` in `Model` will catch exceptions during execution and return error objects per-call.

---

## Example

```py
def greet(name: str) -> str:
    return f"Hello, {name}!"

greet_decl = {
    "name": "greet",
    "description": "Greet a person",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
}

greet_tool = Tool("greet", greet_decl, greet, default_params={"name": "User"})
print(greet_tool.run(name="Phoenix"))  # -> "Hello, Phoenix!"
```
