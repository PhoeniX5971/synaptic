"""A run moves to another model at a turn boundary; agents delegate through tools."""

import asyncio

import pytest

from synaptic import Agent, Ledger, ResponseMem, ToolCall, subagent
from tests.test_execution_recovery import make_model, make_tool


def named(model, name, text=None):
    model.model = name
    if text is not None:
        model.llm.invoke = lambda **kwargs: ResponseMem(text, None)

        async def stream(**kwargs):
            from synaptic import ResponseChunk

            yield ResponseChunk(text, is_final=True)

        model.llm.astream = stream
    return model


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_handoff_mid_run_keeps_session_and_tools(mode):
    tool = make_tool("effect", lambda: "ok")
    first = named(make_model([tool], [ToolCall("effect", {}, id="c1")]), "small")
    second = named(make_model([tool], []), "large", "final answer")
    bound = []
    second.bind_tools = lambda tools: bound.extend(t.name for t in tools)
    ledger = Ledger()
    agent = Agent(first, tools=[tool], ledger=ledger)
    agent.on("tool_end", lambda call, result: agent.handoff(second))

    async def drive():
        if mode == "async":
            return (await agent.arun("go")).message
        return [chunk.text async for chunk in agent.astream("go")][-1]

    answer = agent.run("go").message if mode == "sync" else asyncio.run(drive())
    assert answer == "final answer" and agent.model is second
    assert second.history is agent.session.history and bound == ["effect"]
    kinds = [entry["kind"] for entry in ledger.entries]
    assert kinds == [
        "run_start", "input", "model", "tool_start", "tool_end", "handoff", "model", "run_end",
    ]
    assert ledger.of("handoff")[0]["source"] == "small"
    assert [entry["model"] for entry in ledger.of("model")] == ["small", "large"]
    assert len(agent.session.history.entries) == 3


def test_handoff_rejects_active_model_and_idle_handoff_applies_next_run():
    agent = Agent(named(make_model([], []), "a", "from a"))
    busy = named(make_model([], []), "busy")
    busy._operation_owner = object()
    agent.handoff(busy)
    with pytest.raises(RuntimeError, match="active Model"):
        agent.run("go")
    agent.session.reset()
    agent.handoff(named(make_model([], []), "b", "from b"))
    assert agent.run("again").message == "from b"


@pytest.mark.parametrize("sync", [True, False])
def test_agent_delegates_to_subagent(sync):
    worker = Agent(named(make_model([], []), "worker", "researched"))
    tool = worker.as_tool("research", "Research a topic", sync=sync)
    lead = Agent(
        make_model([tool], [ToolCall("research", {"task": "find x"}, id="d1")]),
        tools=[tool],
        max_turns=1,
    )
    with pytest.raises(RuntimeError, match="max_turns"):
        lead.run("go") if sync else asyncio.run(lead.arun("go"))
    assert lead.session.history.entries[-1].tool_results[0]["result"] == "researched"
    assert worker.session.history.entries[0].message == "find x"
    assert worker.session.id != lead.session.id and tool.metadata["parallel"] is False


def test_factory_subagents_run_in_parallel_with_fresh_sessions():
    made = []

    def factory():
        model = named(make_model([], []), "worker")

        async def slow(prompt, **kwargs):
            await asyncio.sleep(0.05)
            return ResponseMem(f"done {len(made)}", None)

        model.ainvoke = slow
        made.append(Agent(model))
        return made[-1]

    tool = subagent("work", "Do one unit of work", factory)
    calls = [ToolCall("work", {"task": f"t{i}"}, id=f"d{i}") for i in range(3)]
    lead = Agent(make_model([tool], calls), tools=[tool], max_turns=1, max_parallel_tools=3)

    async def drive():
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(RuntimeError, match="max_turns"):
            await lead.arun("go")
        return loop.time() - started

    assert asyncio.run(drive()) < 0.12 and len(made) == 3
    assert len({agent.session.id for agent in made}) == 3
    assert [r["call_id"] for r in lead.session.history.entries[-1].tool_results] == ["d0", "d1", "d2"]


def test_cancelling_the_parent_cancels_the_delegate():
    async def drive():
        model = named(make_model([], []), "worker")
        state = {}

        async def hang(prompt, **kwargs):
            state["started"] = True
            await asyncio.sleep(10)

        model.ainvoke = hang
        worker = Agent(model)
        tool = worker.as_tool("work", "Work")
        lead = Agent(make_model([tool], [ToolCall("work", {"task": "t"}, id="d1")]), tools=[tool])
        run = asyncio.create_task(lead.arun("go"))
        while not state:
            await asyncio.sleep(0)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        return worker.session.outcome, lead.session.outcome

    assert asyncio.run(drive()) == ("cancelled", "cancelled")


def test_handoff_between_real_models_moves_tools_history_and_instructions():
    from synaptic import Model, Provider, Tool

    declaration = {"name": "effect", "description": "", "parameters": {"type": "object"}}
    tool = Tool("effect", declaration, lambda: "ok", add_to_registry=False)
    first = Model(Provider.OPENAI, "small", api_key="unused", instructions="Be brief.")
    second = Model(Provider.CLAUDE, "large", api_key="unused")
    agent = Agent(first, tools=[tool])
    assert [t.name for t in first.llm.synaptic_tools] == ["effect"]
    agent.handoff(second)
    agent._prepare_turn()
    assert agent.model is second and second.history is agent.session.history
    assert [t.name for t in second.llm.synaptic_tools] == ["effect"]
    assert first.llm.synaptic_tools == [] and second.instructions == "Be brief."
