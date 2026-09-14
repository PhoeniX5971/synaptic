"""Live signal mode test against local ollama (phi4-mini via UniversalOpenAI)."""
import asyncio
from synaptic import Agent, Provider, SignalMode
from synaptic.agent.events import EventBus
from synaptic.core.model import Model
from synaptic.core.tool import Tool, ToolCall


def add(a: int, b: int) -> int:
    return a + b


ADD_TOOL = Tool(
    name="add",
    declaration={
        "name": "add",
        "description": "Add two integers and return the sum.",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
    },
    function=add,
    add_to_registry=False,
)


def test_signal_mode_tool_call():
    asyncio.run(_test_signal_mode_tool_call())


async def _test_signal_mode_tool_call():
    model = Model(
        provider=Provider.UNIVERSAL_OPENAI,
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        max_tokens=512,
        temperature=0.0,
    )

    signals = EventBus()
    fired = {"started": [], "done": [], "result": [], "text": []}

    arg_deltas = []
    signals.on("ToolCallStarted", lambda e: fired["started"].append(e.name))
    signals.on("ToolCallDone", lambda e: fired["done"].append(e.call.name))
    signals.on("ToolCallResult", lambda e: fired["result"].append(e.result))
    signals.on("TextDelta", lambda e: fired["text"].append(e.text))
    signals.on("ToolCallArgsDelta", lambda e: arg_deltas.append(e.delta))

    agent = Agent(
        model=model,
        tools=[ADD_TOOL],
        signals=signals,
        signal_mode=SignalMode.AUTO,
        max_turns=3,
    )

    async for _ in agent.astream("What is 7 plus 5? Use the add tool."):
        pass

    assert fired["started"], "ToolCallStarted never fired"
    assert "add" in fired["started"], f"expected 'add' in {fired['started']}"
    assert fired["done"], "ToolCallDone never fired"
    assert fired["result"], "ToolCallResult never fired"
    print(f"\nstarted={fired['started']} done={fired['done']} result={fired['result']}")
