"""Executable reproductions of reviewed core defects (BUG-004 through BUG-007)."""

import asyncio
from types import SimpleNamespace

import pytest

from synaptic.core import _runner
from synaptic.core.base import History, ResponseChunk
from synaptic.core.model import Model
from synaptic.core.tool import Tool, ToolCall
from synaptic.run import Retry


def test_async_usage_survives_aggregation():
    async def stream(**kwargs):
        yield ResponseChunk(text="answer")
        yield ResponseChunk(text="answer", is_final=True, input_tokens=12, output_tokens=3)

    model = SimpleNamespace(
        llm=SimpleNamespace(astream=stream), autorun=False, automem=True, history=History()
    )
    result = asyncio.run(_runner.ainvoke(model, "prompt"))
    assert (result.input_tokens, result.output_tokens) == (12, 3)


def test_stream_cannot_replay_visible_output():
    class Flaky:
        attempts = 0

        async def astream(self, *args, **kwargs):
            self.attempts += 1
            yield "first"
            if self.attempts == 1:
                raise ConnectionError("lost")
            yield "last"

    async def run():
        model = Flaky()
        seen = []
        with pytest.raises(ConnectionError, match="lost"):
            async for chunk in Retry(model, delay=0).astream("prompt"):
                seen.append(chunk)
        assert seen == ["first"]
        assert model.attempts == 1

    asyncio.run(run())


def test_tool_runner_resolves_future():
    async def run():
        future = asyncio.get_running_loop().create_future()
        future.set_result(42)
        tool = Tool("future", {}, lambda: future, add_to_registry=False)
        result = await _runner.run_tools_async([tool], [], [ToolCall("future", {})])
        assert result == [{"name": "future", "result": 42}]

    asyncio.run(run())


def test_binding_deduplicates_incoming_batch():
    model = Model.__new__(Model)
    model.tools = []
    model.llm = SimpleNamespace(bound_tools=[], synaptic_tools=[], _invalidate_tools=lambda: None)
    tool = Tool("echo", {}, add_to_registry=False)
    model.bind_tools([tool, tool])
    assert [t.name for t in model.tools] == ["echo"]
