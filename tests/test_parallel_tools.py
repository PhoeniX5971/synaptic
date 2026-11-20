"""Parallel-safe tools overlap; receipts, order and failure evidence stay truthful."""

import asyncio
import time

import pytest

from synaptic import Agent, EventBus, ResponseMem, Tool, ToolCall, autotool
from tests.test_execution_recovery import make_model


def napper(name, log, parallel=True, fail=False):
    async def nap():
        log.append(("start", name))
        await asyncio.sleep(0.05)
        if fail:
            raise LookupError(name)
        log.append(("end", name))
        return name

    meta = {"parallel": True} if parallel else None
    return Tool(name, {"parameters": {"type": "object"}}, nap, add_to_registry=False, metadata=meta)


def run(tools, limit, events=None):
    calls = [ToolCall(tool.name, {}, id=tool.name) for tool in tools]
    agent = Agent(make_model(tools, calls), tools=tools, max_parallel_tools=limit, events=events)
    memory = ResponseMem("", None, tool_calls=calls)
    started = time.monotonic()
    try:
        asyncio.run(agent._atool_results(calls, memory))
    finally:
        elapsed = time.monotonic() - started
    return memory, elapsed


def test_marked_tools_overlap_and_keep_call_order():
    log, ended, bus = [], [], EventBus(errors="raise")
    bus.on("tool_end", lambda call, result: ended.append((call.id, result["latency_ms"])))
    memory, elapsed = run([napper(n, log) for n in "abc"], 3, bus)
    assert elapsed < 0.12 and [kind for kind, _ in log[:3]] == ["start"] * 3
    assert [r["call_id"] for r in memory.tool_results] == ["a", "b", "c"]
    assert [call_id for call_id, _ in ended] == ["a", "b", "c"]
    assert all(40 <= ms < 100 for _, ms in ended)


@pytest.mark.parametrize("limit, parallel", [(1, True), (3, False)])
def test_sequential_without_limit_or_marker(limit, parallel):
    log = []
    _, elapsed = run([napper(n, log, parallel) for n in "ab"], limit)
    assert elapsed >= 0.1 and log[1] == ("end", "a")


def test_concurrency_cap_and_unmarked_barrier():
    log = []
    run([napper("a", log), napper("b", log), napper("c", log)], 2)
    assert log.index(("start", "c")) > min(log.index(("end", "a")), log.index(("end", "b")))
    log.clear()
    run([napper("a", log), napper("x", log, parallel=False), napper("b", log)], 3)
    assert log.index(("end", "a")) < log.index(("start", "x")) < log.index(("start", "b"))


def test_sibling_receipts_survive_a_failed_call():
    log = []
    tools = [napper("a", log), napper("b", log, fail=True), napper("c", log)]
    calls = [ToolCall(tool.name, {}, id=tool.name) for tool in tools]
    agent = Agent(make_model(tools, calls), tools=tools, max_parallel_tools=3)
    memory = ResponseMem("", None, tool_calls=calls)
    with pytest.raises(LookupError):
        asyncio.run(agent._atool_results(calls, memory))
    assert [r["call_id"] for r in memory.tool_results] == ["a", "c"]
    statuses = {key: value["status"] for key, value in memory.tool_outcomes.items()}
    assert statuses == {"a": "succeeded", "b": "uncertain", "c": "succeeded"}
    assert memory.extensions["incomplete"] is True


def test_autotool_marker():
    @autotool("fetch", autobind=False, parallel=True)
    def fetch(url: str) -> str:
        return url

    @autotool("write", autobind=False)
    def write(path: str) -> str:
        return path

    assert fetch.metadata == {"parallel": True} and write.metadata == {}
