# Examples

This folder contains runnable examples. The core patterns are below.

## One Model Call

```python
from synaptic import Model, Provider

model = Model(provider=Provider.OPENAI, model="gpt-4o-mini", api_key="...")
print(model.invoke("Hello").message)
```

## Tool And Agent

```python
from synaptic import Agent, Model, Provider, autotool

@autotool("Multiply two numbers")
def multiply(a: int, b: int) -> int:
    return a * b

model = Model(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    api_key="...",
    tools=[multiply],
)

agent = Agent(model)
res = agent.run("Multiply 6 by 7 and answer with the result")
print(res.message)
```

## Streaming Events

```python
agent.on("text_delta", lambda text: print(text, end=""))
agent.on("tool_start", lambda call: print("tool:", call.name))
```

See `showcase.py` for a fuller example.
