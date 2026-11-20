"""The ledger records runs in order and supports fork, serialization and replay."""

import asyncio
import json
import warnings

import pytest

from synaptic import Agent, Ledger, ToolCall
from synaptic.run import areplay, compare, replay
from tests.test_execution_recovery import make_model, make_tool


def agent_with(ledger, fail=False):
    def effect():
        if fail:
            raise LookupError("boom")
        return "ok"

    tool = make_tool("effect", effect)
    model = make_model([tool], [ToolCall("effect", {}, id="c1")])
    return Agent(model, tools=[tool], max_turns=1, ledger=ledger)


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_records_a_run_in_order(mode):
    stored, ledger = [], Ledger()
    ledger.sink = stored.append
    agent = agent_with(ledger)

    async def drive():
        if mode == "async":
            await agent.arun("go")
        else:
            async for _ in agent.astream("go"):
                pass

    with pytest.raises(RuntimeError, match="max_turns"):
        agent.run("go") if mode == "sync" else asyncio.run(drive())
    kinds = [entry["kind"] for entry in ledger.entries]
    assert kinds == ["run_start", "input", "model", "tool_start", "tool_end", "run_end"]
    assert [entry["seq"] for entry in ledger.entries] == list(range(6))
    assert {entry["session_id"] for entry in ledger.entries} == {agent.session.id}
    model, end, done = ledger.of("model")[0], ledger.of("tool_end")[0], ledger.of("run_end")[0]
    assert model["tool_calls"] == [{"id": "c1", "name": "effect", "args": {}}]
    assert isinstance(model["latency_ms"], int)
    assert end["call_id"] == "c1" and end["result"] == "ok" and end["latency_ms"] >= 0
    assert done["outcome"] == "exhausted"
    assert stored == ledger.entries == json.loads(json.dumps(ledger.to_list()))


def test_entries_are_detached_and_fork_is_independent():
    ledger = Ledger()
    args = {"path": "a"}
    ledger.append("tool_start", args=args)
    args["path"] = "changed"
    ledger.entries[0]["kind"] = "edited"
    assert ledger.entries[0] == {**ledger.entries[0], "kind": "tool_start", "args": {"path": "a"}}
    ledger.append("tool_end")
    fork = ledger.fork(0)
    fork.append("other")
    assert [e["kind"] for e in fork.entries] == ["tool_start", "other"] and len(ledger) == 2
    assert Ledger.from_list(ledger.to_list()).entries == ledger.entries


def test_failed_run_and_failing_sink_do_not_lose_the_record():
    def sink(entry):
        raise OSError("disk full")

    ledger = Ledger(sink=sink)
    agent = agent_with(ledger, fail=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(LookupError):
            agent.run("go")
    assert [e["kind"] for e in ledger.entries][-2:] == ["tool_start", "run_end"]
    assert ledger.of("run_end")[0]["outcome"] == "failed"
    assert any("Ledger sink failed: OSError" in str(item.message) for item in caught)


def test_detach_stops_recording():
    ledger = Ledger()
    agent = agent_with(None)
    with ledger.attach(agent):
        assert agent.ledger is ledger
    with pytest.raises(RuntimeError):
        agent.run("go")
    assert len(ledger) == 0 and agent.ledger is None


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_replay_and_compare(mode):
    original = Ledger()
    with pytest.raises(RuntimeError):
        agent_with(original).run("go")

    def effect():
        return "ok"

    tool = make_tool("effect", effect)
    other = Agent(make_model([tool], []), tools=[tool], max_turns=1)
    fresh = replay(original, other) if mode == "sync" else asyncio.run(areplay(original, other))
    assert fresh.of("input")[0]["prompt"] == "go" and other.ledger is None
    report = compare(original, fresh)
    assert report["before"]["tool_calls"] == ["effect"] and report["after"]["tool_calls"] == []
    assert report["same_tool_sequence"] is False and report["delta"]["model_calls"] == 0
    assert report["after"]["outcomes"] == ["completed"]
