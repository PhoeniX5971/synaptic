"""Control BUG-012–014 through public Model, Retry and history APIs."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from synaptic import (
    Model,
    Provider,
    History,
    ResponseMem,
    ResponseChunk,
    Tool,
    ToolCall,
    complete_tool_call,
)
from synaptic.run import Retry


def model_with(adapter, **kwargs):
    adapter._invalidate_tools = lambda: None
    adapter.synaptic_tools = getattr(adapter, "synaptic_tools", [])
    adapter.bound_tools = list(adapter.synaptic_tools)
    with patch("synaptic.core.model.build_adapter", return_value=adapter):
        return Model(Provider.OPENAI, "fixture", **kwargs)


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
@pytest.mark.parametrize("automem", [False, True])
def test_model_retry_automem_before_output(mode, automem):
    async def run():
        attempts = []

        async def stream(**kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionError("before output")
            yield ResponseChunk("ok", is_final=True)

        def invoke(**kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionError("before output")
            return ResponseMem("ok", None)

        model = model_with(SimpleNamespace(astream=stream, invoke=invoke), automem=automem)
        retry = Retry(model, max_retries=1, delay=0, jitter=0)
        if mode == "sync":
            assert retry.invoke("hello").message == "ok"
        elif mode == "async":
            assert (await retry.ainvoke("hello")).message == "ok"
        else:
            assert [chunk.text async for chunk in retry.astream("hello")] == ["ok"]
        assert len(attempts) == 2
        assert [m.message for m in model.history.entries] == (
            ["hello", "ok"] if automem else []
        )
        assert len(model.history.failed_attempts) == 1

    asyncio.run(run())


def test_context_budget_after_result_growth():
    def count(memory):
        return len(memory.message) + sum(
            len(r.get("result", "")) for r in getattr(memory, "tool_results", [])
        )

    history = History(token_budget=20, token_counter=count, on_truncate=lambda dropped: None)
    tool = Tool(
        "large", {"parameters": {"type": "object"}}, lambda: "x" * 1000, add_to_registry=False
    )
    seen = []

    def invoke(**kwargs):
        seen.append(sum(count(m) for m in history.effective_mems()))
        return (
            ResponseMem("", None, tool_calls=[ToolCall("large", {}, id="large-id")])
            if len(seen) == 1
            else ResponseMem("ok", None)
        )

    model = model_with(
        SimpleNamespace(invoke=invoke, synaptic_tools=[tool]),
        history=history,
        autorun=True,
        automem=True,
    )
    receipt = model.invoke("go")
    assert sum(count(m) for m in history.entries) <= 20
    model.invoke(None)
    assert seen[-1] <= 20
    assert receipt in history.recovery
    assert receipt.tool_results[0]["result"] == "x" * 1000


def test_contradictory_mapping_rejected_atomically():
    memory = ResponseMem("", None, tool_calls=[ToolCall("effect", {}, id="right")])
    with pytest.raises(ValueError):
        complete_tool_call(memory, {"right": {"call_id": "wrong", "result": "original"}})
    assert memory.tool_results == [] and memory.extensions == {}
