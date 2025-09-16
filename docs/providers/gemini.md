# Provider — Gemini Adapter

**File:** `providers/gemini/model.py`

Adapter that wraps Google GenAI (Gemini) SDK and converts internal `Memory`
objects and `Tool` declarations into Gemini `types.Content` and `types.Tool`
objects.

---

## Class: `GeminiAdapter`

**Type:** class — implements `BaseModel` interface

**Constructor**

```py
GeminiAdapter(model: str, history: History, temperature: float = 0.8, tools: list | None = None)
```

**Behavior**

- Converts registered `Tool` objects to Gemini `types.Tool` via `_convert_tools`.
- Serializes `History.MemoryList` into Gemini `types.Content` objects in `to_contents`.
  - Each memory `parts` includes the `message` and a "(Created at: ...)" part.
  - For `ResponseMem`, `tool_calls` and `tool_results` are appended as parts.
- When `invoke(prompt)` is called:
  - Builds `contents = self.to_contents()` and appends the incoming prompt as user content.
  - Calls `self.client.models.generate_content(model=..., contents=..., config=...)`.
  - Parses `response.candidates[0]` to build `message` and `tool_calls` (from `part.function_call`).
  - Returns `ResponseMem(message=..., created=..., tool_calls=...)`.

**Notes / Environment**

- The implementation expects `google.genai` to be installed and environment credentials to be configured as required by the Google GenAI SDK.

- Add `.env` usage or authentication docs as needed for deployment.

---

## Example

```py
from synaptic.providers.gemini.model import GeminiAdapter
from synaptic.core.base.memory import History

history = History()
adapter = GeminiAdapter(model="gemini-2.5-pro", history=history)
res = adapter.invoke("Hello Gemini!")
print(res.message)
print(res.tool_calls)
```

**Important**

- Provider SDKs may change APIs — verify compatibility with your `google.genai` version.
- Network errors and API errors should be handled by your outer workflow.
