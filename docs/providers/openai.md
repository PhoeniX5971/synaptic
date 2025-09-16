# Provider — OpenAI Adapter

**File:** `providers/openai/model.py`

Adapter wrapping the OpenAI SDK (chat completions / function calling).
Converts internal `Tool` objects to OpenAI `functions` definitions and parses
responses into `ResponseMem`.

---

## Class: `OpenAIAdapter`

**Type:** class — implements `BaseModel` interface

**Constructor**

```py
OpenAIAdapter(model: str, tools: List[Tool] | None = None)
```

**Behavior**

- `_convert_tools()` transforms the `Tool` declaration into OpenAI's  
   `functions` format:
  ```py
  {"name": t.name, "description": t.declaration.get("description", ""), "parameters": t.declaration.get("parameters", {})}
  ```
- `invoke(prompt: str, **kwargs)` calls:
  ```py
    response = self.client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": prompt}],
        functions=self._convert_tools() if self.tools else None,
        **kwargs
    )
  ```
- Parses `choice = response.choices[0]` and:
  - Extracts `message = choice.message.get("content")`
  - If `choice.message.get("function_call")` exists, decodes `arguments` as JSON and yields `tool_calls`.
- Returns `ResponseMem(message=..., created=..., tool_calls=...)`.

**Environment**

- Expects the OpenAI client to be configured (API key in env or equivalent).
- Check your openai SDK version and adapt the call shape if necessary.

---

## Example

```py
from synaptic.providers.openai.model import OpenAIAdapter
from synaptic.core.tool import Tool

adapter = OpenAIAdapter(model="gpt-4o-mini", tools=[/* Tool instances */])
res = adapter.invoke("If needed, call add with a=2 and b=3")
print(res.message)
print(res.tool_calls)
```

**Notes**

- The code shown is using `client.chat.completions.create` — confirm the correct call for your OpenAI SDK version (names and call shapes differ across releases).
