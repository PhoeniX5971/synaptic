# Synaptic

Synaptic is a small Python framework for building LLM applications with a
provider-neutral `Model`, typed conversation memory, tool/function calling,
multi-turn agents, streaming events, and lightweight workflow helpers.

## Install

```bash
git clone https://github.com/PhoeniX5971/synaptic.git
cd synaptic
pip install -e ".[openai]"
```

Python 3.11+ is the declared floor. The test and typing gate passes on CPython 3.11,
3.12 and 3.13 on Linux x86_64 with the pinned SDKs; installs are checked on 3.13.
Windows, macOS and live provider services are unmeasured.
Core installation (`pip install -e .`) includes no provider SDKs.
Choose extras: `openai`, `deepseek`, `claude`, `gemini`, `vertex`, `together`, `xai`,
`mcp`, or `all`. Pass credentials explicitly or configure the chosen SDK environment;
Synaptic does not load `.env` files. See [runtime contracts](docs/contracts.md),
[migrations](docs/migration.md), and [measured support and limitations](docs/support.md).

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

agent = Agent(model, tools=[add], max_turns=5)
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
- `EventBus`: subscribe to agent text/tool/step events, with tool and model latency.
- `Flow` / `Pipeline`: lightweight sync/async workflow helpers.
- `Ledger`: append-only record of runs with storage sink, fork, replay and compare.
- Extensions: `FinalAnswer` (structured result that ends a run), `Todos` (plan with
  required items) and `Skills` (instructions and tools loaded on demand), or your own.
- Agent control: task state, context compiler, model handoff, subagents, parallel tools,
  and tool failures returned to the model on request.

## Documentation

Start with [`docs/README.md`](docs/README.md). The docs cover every public
surface needed to use Synaptic without prior knowledge.

## License

GNU General Public License v3.0. See the repository license for details.

## Contributing and release status

See [CONTRIBUTING](CONTRIBUTING.md), [security reporting](SECURITY.md),
[release guidance](docs/releasing.md) and [changelog](CHANGELOG.md). The
[tracked qualification workflow](.github/workflows/qualification.yml) runs the frozen
gates, but it currently fails to start on GitHub and has never completed a run, so all
results are local. Try the credential-free
[offline example](examples/offline.py) for tools and serialized recovery receipts.
