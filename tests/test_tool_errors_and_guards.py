"""Returned tool errors, failure events, one loop for all run modes, and guards."""

import asyncio

import pytest

from synaptic import Agent, EventBus, History, Ledger, ResponseChunk, ResponseMem, Tool, ToolCall
from synaptic import UserMem, autotool
from synaptic.run import replay
from tests.test_execution_recovery import make_model, make_tool


def scripted(tools, *responses):
    """A model that answers each call with the next scripted tool calls or text."""
    model, queue = make_model(tools, []), list(responses)
    model.seen = []

    def following():
        return queue.pop(0) if queue else "done"

    def memory(item):
        return ResponseMem(item, None) if isinstance(item, str) else ResponseMem("", None, tool_calls=item)

    async def stream(**kwargs):
        model.seen.append(kwargs)
        item = following()
        if isinstance(item, str):
            yield ResponseChunk(item)
            yield ResponseChunk(item, is_final=True)
        else:
            for call in item:
                yield ResponseChunk("", function_call=call)
            yield ResponseChunk("", is_final=True)

    model.llm.invoke = lambda **kwargs: memory(following())
    model.llm.astream = stream
    return model


def boom():
    raise LookupError("boom")


async def drive(agent, mode, prompt="go", **kwargs):
    if mode == "sync":
        return agent.run(prompt, **kwargs)
    if mode == "async":
        return await agent.arun(prompt, **kwargs)
    async for _ in agent.astream(prompt, **kwargs):
        pass
    return agent.session.history.entries[-1]


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_returned_error_reaches_the_model_and_the_run_continues(mode):
    tool, ended, ledger = make_tool("effect", boom), [], Ledger()
    model = scripted([tool], [ToolCall("effect", {}, id="c1")], "recovered")
    agent = Agent(model, tools=[tool], tool_errors="return", ledger=ledger)
    agent.on("tool_end", lambda call, result: ended.append(result))
    final = asyncio.run(drive(agent, mode))
    exchange = agent.session.history.entries[1]
    assert final.message == "recovered" and agent.session.outcome == "completed"
    assert exchange.tool_results == [{"name": "effect", "call_id": "c1", "error": "LookupError: boom"}]
    assert exchange.tool_outcomes["c1"]["status"] == "failed" and "incomplete" not in exchange.extensions
    assert ended[0]["error"] == "LookupError: boom" and ended[0]["latency_ms"] >= 0
    assert [e["kind"] for e in ledger.entries if e["kind"].startswith("tool")] == ["tool_start", "tool_end"]


@pytest.mark.parametrize("agent_mode, tool_mode, returned", [("raise", "return", True), ("return", "raise", False)])
def test_per_tool_setting_overrides_the_agent(agent_mode, tool_mode, returned):
    tool = Tool("effect", {"parameters": {"type": "object"}}, boom, add_to_registry=False, metadata={"errors": tool_mode})
    agent = Agent(scripted([tool], [ToolCall("effect", {}, id="c1")]), tools=[tool], tool_errors=agent_mode)
    if returned:
        assert agent.run("go").message == "done"
    else:
        with pytest.raises(LookupError):
            agent.run("go")

    @autotool("x", autobind=False, errors="return", parallel=True)
    def marked() -> str:
        return ""

    assert marked.metadata == {"parallel": True, "errors": "return"}
    with pytest.raises(ValueError, match="tool_errors"):
        Agent(scripted([], "x"), tool_errors="ignore")


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_raising_tool_emits_tool_error_with_latency(mode):
    tool, seen = make_tool("effect", boom), []
    agent = Agent(scripted([tool], [ToolCall("effect", {}, id="c1")]), tools=[tool])
    agent.on("tool_error", lambda call, result: seen.append((call.id, result)))
    with pytest.raises(LookupError):
        asyncio.run(drive(agent, mode))
    assert seen[0][0] == "c1" and seen[0][1]["error"] == "LookupError: boom"
    assert seen[0][1]["latency_ms"] >= 0 and agent.session.history.entries[-1].tool_results == []
    assert agent.session.history.entries[-1].tool_outcomes["c1"]["status"] == "uncertain"


def test_parallel_batch_returns_errors_in_call_order():
    async def ok():
        await asyncio.sleep(0.01)
        return "ok"

    async def bad():
        raise LookupError("boom")

    meta = {"parallel": True}
    tools = [
        Tool(name, {"parameters": {"type": "object"}}, fn, add_to_registry=False, metadata=meta)
        for name, fn in (("a", ok), ("b", bad), ("c", ok))
    ]
    calls = [ToolCall(tool.name, {}, id=tool.name) for tool in tools]
    agent = Agent(scripted(tools, calls, "after"), tools=tools, tool_errors="return", max_parallel_tools=3)
    assert asyncio.run(agent.arun("go")).message == "after"
    exchange = agent.session.history.entries[1]
    assert [r["call_id"] for r in exchange.tool_results] == ["a", "b", "c"]
    assert [v["status"] for v in exchange.tool_outcomes.values()] == ["succeeded", "failed", "succeeded"]


def test_arun_is_the_stream_drained():
    def run(mode):
        deltas, bus = [], EventBus(errors="raise")
        bus.on("text_delta", deltas.append)
        tool = make_tool("effect", lambda: "ok")
        agent = Agent(scripted([tool], [ToolCall("effect", {}, id="c1")], "answer"), tools=[tool], events=bus)
        final = asyncio.run(drive(agent, mode))
        history = [(type(m).__name__, m.message) for m in agent.session.history.entries]
        return final.message, deltas, history, agent.session.outcome

    assert run("async") == run("stream") == (
        "answer",
        ["answer"],
        [("UserMem", "go"), ("ResponseMem", ""), ("ResponseMem", "answer")],
        "completed",
    )
    assert run("sync")[2:] == run("async")[2:]


def test_nested_run_on_a_shared_model_is_refused():
    model = scripted([], "inner")
    worker = Agent(model)
    tool = worker.as_tool("work", "Work", sync=True)
    model.llm.invoke = lambda **kwargs: ResponseMem("", None, tool_calls=[ToolCall("work", {"task": "t"}, id="d1")])
    lead = Agent(model, tools=[tool])
    with pytest.raises(RuntimeError, match="own Model"):
        lead.run("go")
    assert lead.session.history.entries[0].message == "go" and model._agent_run is None
    model.llm.invoke = lambda **kwargs: ResponseMem("free again", None)
    assert Agent(model).run("go").message == "free again"


def test_parent_abort_event_reaches_the_delegate():
    async def run():
        abort = asyncio.Event()
        worker_model = scripted([], "researched")
        tool = Agent(worker_model).as_tool("work", "Work")
        lead = Agent(scripted([tool], [ToolCall("work", {"task": "t"}, id="d1")], "final"), tools=[tool])
        await lead.arun("go", abort=abort)
        return worker_model.seen[0]["abort"] is abort

    assert asyncio.run(run())


def test_compiler_output_is_validated():
    history = History([UserMem("hello", None)])
    for bad in (lambda mems: [], lambda mems: ["text"]):
        history.compiler = bad
        with pytest.raises(TypeError, match="Memory objects"):
            history.effective_mems()
    history.compiler = lambda mems: [ResponseMem("", None, tool_calls=[ToolCall("t", {}, id="c1")])]
    with pytest.raises(ValueError, match="without its results"):
        history.effective_mems()


def test_ledger_records_media_references_and_replay_resends_them():
    ledger, received = Ledger(), []
    Agent(scripted([], "seen"), ledger=ledger).run("look", images=["https://example.test/a.png", b"raw"])
    assert ledger.of("input")[0]["images"] == ["https://example.test/a.png"]
    other = Agent(scripted([], "again"))
    original = other._add_user
    other._add_user = lambda prompt, role, images=None, audio=None: (
        received.append(images),
        original(prompt, role, images, audio),
    )
    replay(ledger, other)
    assert received == [["https://example.test/a.png"]]


def test_claude_binds_a_tool_declared_without_a_name():
    from synaptic import Model, Provider

    tool = make_tool("effect", lambda: "ok")
    model = Model(Provider.CLAUDE, "fixture", api_key="unused", tools=[tool])
    assert model.llm.claude_tools[0]["name"] == "effect"
