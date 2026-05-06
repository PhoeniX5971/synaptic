# Synaptic

Synaptic is a small Python framework for building LLM applications with a
provider-neutral `Model`, typed conversation memory, tool/function calling,
multi-turn agents, streaming events, and lightweight workflow helpers.

## Install

```bash
git clone https://github.com/PhoeniX5971/synaptic.git
cd synaptic
pip install -e .
```

Install the SDKs for the providers you plan to use. For example, OpenAI-backed
models need the OpenAI SDK and an API key passed to `Model(api_key=...)` or
configured in your environment.

## First Model Call

```python
from synaptic import Model, Provider

model = Model(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    api_key="...",
    instructions="Answer briefly.",
)

res = model.invoke("What is Synaptic?")
print(res.message)
```

## Tools

```python
from synaptic import Model, Provider, autotool

@autotool("Add two integers")
def add(a: int, b: int) -> int:
    return a + b

model = Model(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    api_key="...",
    tools=[add],
    autorun=True,
)

res = model.invoke("Use the add tool for 2 + 3")
print(res.tool_calls)
print(res.tool_results)
```

## Agent Loop

Use `Agent` when tools should feed back into the model until the answer is
finished.

```python
from synaptic import Agent

agent = Agent(model, max_turns=5)
res = agent.run("Calculate 2 + 3 and explain the result")
print(res.message)
```

## Streaming

```python
async for chunk in model.astream("Write a haiku"):
    print(chunk.text, end="")
```

## Main Concepts

- `Model`: one provider/model configuration.
- `Agent`: multi-turn tool loop around a model.
- `Tool` / `autotool`: expose Python functions to providers.
- `ToolRegistry`: isolate tools per test, user, tenant, or session.
- `History`: sliding-window memory used as provider context.
- `Session`: state and history container for agent runs.
- `EventBus`: subscribe to agent text/tool/step events.
- `Flow` / `Pipeline`: lightweight sync/async workflow helpers.

## Documentation

Start with [`docs/README.md`](docs/README.md). The docs cover every public
surface needed to use Synaptic without prior knowledge.

## License

GNU General Public License v3.0. See the repository license for details.
