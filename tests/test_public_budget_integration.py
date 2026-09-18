import asyncio
import json
from types import SimpleNamespace
import pytest
from synaptic import Agent, History, Session, ResponseMem, ResponseChunk, Tool, ToolCall
from tests.test_public_integration import model_with


def count(memory):
    return len(memory.message) + sum(
        len(item.get("result", "")) for item in getattr(memory, "tool_results", [])
    )


@pytest.mark.parametrize("kind", ["model", "agent"])
@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
@pytest.mark.parametrize("partial", [False, True])
def test_growth_projects_context_but_preserves_recovery(kind, mode, partial):
    async def run():
        seen = []
        history = History(
            token_budget=20, token_counter=count, on_truncate=lambda dropped: None
        )

        def fail():
            raise ValueError("second effect failed")

        tools = [
            Tool(
                "large",
                {"parameters": {"type": "object"}},
                lambda: "x" * 1000,
                add_to_registry=False,
            ),
            Tool("fail", {"parameters": {"type": "object"}}, fail, add_to_registry=False),
        ]

        def invoke(**kwargs):
            seen.append(sum(count(m) for m in model.history.effective_mems()))
            calls = [ToolCall("large", {}, id="large-id")]
            if partial:
                calls.append(ToolCall("fail", {}, id="fail-id"))
            return (
                ResponseMem("", None, tool_calls=calls)
                if len(seen) == 1
                else ResponseMem("ok", None)
            )

        async def stream(**kwargs):
            response = invoke(**kwargs)
            for call in response.tool_calls:
                yield ResponseChunk("", function_call=call)
            yield ResponseChunk(response.message, is_final=True)

        model = model_with(
            SimpleNamespace(invoke=invoke, astream=stream, synaptic_tools=tools),
            history=history,
            automem=True,
            autorun=True,
        )
        subject = (
            Agent(model, tools=tools, session=Session(history=history))
            if kind == "agent"
            else model
        )

        async def drive():
            if mode == "sync":
                return subject.run("go") if kind == "agent" else model.invoke("go")
            if mode == "async":
                return (
                    await subject.arun("go") if kind == "agent" else await model.ainvoke("go")
                )
            return [chunk async for chunk in subject.astream("go")]

        if partial:
            with pytest.raises(ValueError):
                await drive()
        else:
            await drive()
        assert sum(count(m) for m in history.entries) <= 20
        restored = History.from_dict(json.loads(json.dumps(history.to_dict())))
        receipt = restored.recovery[0]
        assert receipt.tool_results[0]["result"] == "x" * 1000
        model.history = restored
        if partial:
            before = len(seen)
            with pytest.raises(ValueError, match="Reconcile"):
                model.invoke(None)
            assert len(seen) == before
            model.supply_results(
                receipt, receipt.tool_results + [{"call_id": "fail-id", "result": "resolved"}]
            )
        restored.token_budget, restored.token_counter, restored.on_truncate = (
            20,
            count,
            lambda lost: None,
        )
        model.invoke(None, autorun=False)
        assert all(cost <= 20 for cost in seen)
        assert receipt.tool_results[0]["result"] == "x" * 1000

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_external_reconciliation_enforces_existing_history_budget(mode):
    async def run():
        history = History(token_budget=20, token_counter=count, on_truncate=lambda lost: None)
        response = ResponseMem(
            "", None, tool_calls=[ToolCall("external", {}, id="external-id")]
        )

        async def stream(**kwargs):
            yield ResponseChunk("", function_call=response.tool_calls[0])
            yield ResponseChunk("", is_final=True)

        model = model_with(
            SimpleNamespace(invoke=lambda **kwargs: response, astream=stream),
            history=history,
            automem=True,
        )
        if mode == "sync":
            proposed = model.propose("go")
        elif mode == "async":
            proposed = await model.apropose("go")
        else:
            _ = [chunk async for chunk in model.astream("go")]
            proposed = model.last_response
        model.supply_results(proposed, {"external-id": {"result": "x" * 1000}})
        assert history.entries == []
        assert history.recovery[0].tool_results == [{"result": "x" * 1000}]

    asyncio.run(run())


def test_direct_completion_and_last_moment_context_growth():
    from synaptic import complete_tool_call

    history = History(token_budget=20, token_counter=count, on_truncate=lambda lost: None)
    response = ResponseMem("", None, tool_calls=[ToolCall("external", {}, id="id")])
    history.add(response)
    complete_tool_call(response, {"id": {"result": "x" * 1000}})
    assert history.entries == [] and response in history.recovery
    text = ResponseMem("small", None)
    history.add(text)
    text.message = "x" * 1000  # Context is checked again immediately before provider entry.
    seen = []

    def invoke(**kwargs):
        seen.append(sum(count(item) for item in history.entries))
        return ResponseMem("ok", None)

    model = model_with(SimpleNamespace(invoke=invoke), history=history)
    model.invoke(None)
    assert seen == [0]


def test_oversized_generation_cancel_retains_unresolved_text():
    abort = asyncio.Event()
    history = History(token_budget=20, token_counter=count, on_truncate=lambda lost: None)

    def invoke(**kwargs):
        abort.set()
        return ResponseMem("x" * 1000, None)

    model = model_with(SimpleNamespace(invoke=invoke), history=history, automem=True)
    with pytest.raises(asyncio.CancelledError):
        model.invoke("go", abort=abort)
    restored = History.from_dict(json.loads(json.dumps(history.to_dict())))
    assert restored.entries == []
    assert restored.recovery[0].extensions["incomplete"]
    assert restored.recovery[0].message == "x" * 1000
