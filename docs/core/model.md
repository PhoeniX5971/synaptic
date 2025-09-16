# Core — Model

**File:** `core/model.py`

`Model` is the high-level orchestrator. It selects the provider adapter
(OpenAI/Gemini), holds registered `Tool` instances, and optionally executes
returned tool calls and stores results in `History`.

---

## Class: `Model`

**Type:** class

**Constructor**

```py
Model(
    provider: Provider,
    model: str,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    tools: List[Tool] | None = None,
    history: History | None = None,
    autorun: bool = False,
    automem: bool = False,
)
```

**Fields**

- `provider: Provider` — provider enum (OPENAI / GEMINI).
- `model: str` — provider model id.
- `temperature: float`
- `max_tokens: int`
- `tools: List[Tool]` — registered tools.
- `history: History` — memory history object.
- `autorun: bool` — when true, `Model.invoke` will execute returned tool calls automatically.
- `automem: bool` — when true, responses are appended to `history`.

**Key methods**

### `_initiate_model() -> BaseModel`

Selects and returns a provider-specific adapter instance (e.g., `OpenAIAdapter` or `GeminiAdapter`).

### `bind_tools(tools: List[Tool])`

Appends tools to the model's tool list.

### `_run_tools(tool_calls: List[dict[str, Any]]) -> List[Any]`

Runs tool calls (map call `name` to registered `Tool.function`) and returns results  
as list of `{"name", "result"}` or `{"name", "error"}`.

**Implementation details**

- Uses `tool_map = {tool.name: tool.function for tool in self.tools}`
- Each call expects a dict with keys `name` and optionally `args` (dict).
- Unregistered tools produce `{"name": name, "error": "Tool not registered"}`.
- Exceptions from tool execution are caught and returned as errors.

### `invoke(prompt: str, autorun: bool = None, automem: bool = None, **kwargs) -> ResponseMem`

1. Instantiates the provider adapter via `_initiate_model`.
2. Calls adapter `invoke(prompt, **kwargs)` to get a `ResponseMem`.
3. If `autorun` is enabled and `memory.tool_calls` is non-empty, runs `_run_tools` and attaches results to `memory.tool_results`.
4. If `automem` is enabled, stores `memory` in `self.history`.
5. Returns the `ResponseMem` instance to the caller.

**Notes**

- Default `autorun` / `automem` are determined by the instance-level flags unless explicitly passed.
- The returned `ResponseMem` will contain `tool_calls` even if `autorun` is false; `tool_results` remain empty unless executed.

---

## Example

```py
from synaptic.core import Model, Provider, Tool, History

add_tool = Tool(
    name="add",
    declaration={"name":"add", "description":"Add numbers", "parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}},
    function=lambda a, b: a + b,
    default_params={"a":0,"b":0},
)

history = History()
model = Model(provider=Provider.GEMINI, model="gemini-2.5-pro", tools=[add_tool], automem=True, autorun=True)

res = model.invoke("Call add with a=2 b=3")
print(res.message)
print(res.tool_calls)
print(res.tool_results)  # will include the executed tool result when autorun=True
```
