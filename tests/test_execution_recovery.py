"""Acceptance regressions for Control's BUG-010/011 readiness probes."""

import asyncio
import json

import pytest

from synaptic import Agent, Tool, ToolCall, ResponseChunk, ResponseMem, History
from synaptic.core import _runner
from tests.test_completion_contract import fixture_model


def make_tool(name, function):
    return Tool(name, {"parameters": {"type": "object"}}, function, add_to_registry=False)


def make_model(tools, calls):
    chunks = [ResponseChunk("", function_call=call) for call in calls] + [
        ResponseChunk("", is_final=True)
    ]
    model = fixture_model(chunks)
    model.blacklist = []
    model.llm.synaptic_tools = tools
    model.llm.invoke = lambda **kwargs: ResponseMem("", None, tool_calls=calls)
    model.invoke = lambda prompt, **kwargs: _runner.invoke(model, prompt, **kwargs)

    async def ainvoke(prompt, **kwargs):
        return await _runner.ainvoke(model, prompt, **kwargs)

    model.ainvoke = ainvoke
    return model


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_agent_preserves_first_receipt_when_second_effect_fails(mode):
    async def run():
        effects = []

        def first():
            effects.append("first")
            return "receipt-first"

        def second():
            effects.append("second-started")
            raise ValueError("second failed")

        tools = [make_tool("first", first), make_tool("second", second)]
        calls = [ToolCall("first", {}, id="first-id"), ToolCall("second", {}, id="second-id")]
        agent = Agent(make_model(tools, calls), tools=tools)
        with pytest.raises(ValueError):
            if mode == "sync":
                agent.run("both")
            elif mode == "async":
                await agent.arun("both")
            else:
                _ = [chunk async for chunk in agent.astream("both")]
        memory = agent.session.history.entries[-1]
        assert memory.tool_results == [
            {"name": "first", "call_id": "first-id", "result": "receipt-first"}
        ]
        restored = History.from_dict(
            json.loads(json.dumps(agent.session.history.to_dict()))
        ).entries[-1]
        assert restored.tool_results == memory.tool_results
        assert restored.tool_outcomes["first-id"]["status"] == "succeeded"
        assert restored.tool_outcomes["second-id"]["status"] == "uncertain"
        assert restored.extensions["incomplete"] is True
        assert agent.session.outcome == "failed" and effects == ["first", "second-started"]

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["model", "agent"])
def test_abort_after_final_yield_prevents_dispatch(kind):
    async def run():
        effects, abort = [], asyncio.Event()
        tools = [make_tool("effect", lambda: effects.append("ran"))]
        model = make_model(tools, [ToolCall("effect", {}, id="effect-id")])
        if kind == "agent":
            agent = Agent(model, tools=tools, max_turns=1)
            source = agent.astream("action", abort=abort)
            history = agent.session.history
        else:
            source = _runner.astream(model, "action", autorun=True, automem=True, abort=abort)
            history = model.history
        with pytest.raises(asyncio.CancelledError):
            async for chunk in source:
                if chunk.is_final:
                    abort.set()
        assert effects == []
        assert history.entries[-1].tool_outcomes["effect-id"]["status"] == "not_started"
        assert history.entries[-1].extensions["incomplete"] is True
        if kind == "agent":
            assert agent.session.outcome == "cancelled"

    asyncio.run(run())
