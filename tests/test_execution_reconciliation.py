"""Persisted recovery, generation cancellation, and cooperative executor boundaries."""

import asyncio
import json

import pytest

from synaptic import History, ResponseChunk, ResponseMem, ToolCall, ToolRegistry
from synaptic.core import _runner
from synaptic.core.base.memory import complete_tool_call
from synaptic.core._tool_execution import run_tools_sync
from tests.test_dispatch_boundaries import drive, setup, outcomes


@pytest.mark.parametrize("kind", ["agent", "model"])
@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_abort_after_first_effect_preserves_result(kind, mode):
    async def run():
        abort, effects = asyncio.Event(), []

        def first():
            effects.append(0)
            abort.set()
            return "receipt"

        subject, model, _, _ = setup(kind, [first, lambda: effects.append(1)])
        with pytest.raises(asyncio.CancelledError):
            await drive(subject, mode, abort=abort)
        memory, statuses = outcomes(model)
        assert statuses == ["succeeded", "not_started"]
        assert memory.tool_results[0]["result"] == "receipt" and effects == [0]

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["agent", "model"])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_abort_set_by_generation_prevents_effect(kind, mode):
    async def run():
        abort, effects = asyncio.Event(), []
        subject, model, _, calls = setup(kind, [lambda: effects.append(0)])

        def invoke(**kwargs):
            abort.set()
            return ResponseMem("", None, tool_calls=calls)

        async def stream(**kwargs):
            yield ResponseChunk("", function_call=calls[0])
            abort.set()
            yield ResponseChunk("", is_final=True)

        model.llm.invoke, model.llm.astream = invoke, stream
        with pytest.raises(asyncio.CancelledError):
            await drive(subject, mode, abort=abort)
        assert outcomes(model)[1] == ["not_started"]
        assert effects == []

    asyncio.run(run())


def test_automem_disabled_retains_last_response_on_failure():
    def fail():
        raise ValueError("effect may have happened")

    _, model, _, _ = setup("model", [lambda: "receipt", fail])
    with pytest.raises(ValueError):
        _runner.invoke(model, "go", autorun=True, automem=False)
    assert model.history.entries == []
    assert model.last_response.tool_results[0]["result"] == "receipt"
    assert model.last_response.tool_outcomes["id1"]["status"] == "uncertain"


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_output_validation_failure_is_uncertain(mode):
    async def run():
        registry = ToolRegistry()

        @registry.autotool()
        def invalid() -> int:
            return "invalid integer"

        subject, model, tools, calls = setup("model", [lambda: "receipt", lambda: 1])
        model.llm.synaptic_tools[1] = registry.all()["invalid"]
        calls[1].name = "invalid"
        with pytest.raises(ValueError):
            await drive(subject, mode)
        assert outcomes(model)[1] == ["succeeded", "uncertain"]

    asyncio.run(run())


def test_reconciliation_cannot_erase_receipts_or_replay_completed_calls():
    def fail():
        raise ValueError("unknown effect")

    subject, model, tools, calls = setup("model", [lambda: "receipt", fail])
    with pytest.raises(ValueError):
        _runner.invoke(model, "go", autorun=True, automem=True)
    restored = History.from_dict(json.loads(json.dumps(model.history.to_dict()))).entries[-1]
    for replacement in [
        [{"call_id": "id0", "result": "changed"}, {"call_id": "id1", "result": "resolved"}],
        ["changed", "resolved"],
        {"id0": "changed", "id1": "resolved"},
    ]:
        with pytest.raises(ValueError, match="already recorded"):
            complete_tool_call(restored, replacement)
    complete_tool_call(
        restored, restored.tool_results + [{"call_id": "id1", "result": "resolved"}]
    )
    assert restored.tool_outcomes["id0"]["status"] == "succeeded"
    assert restored.tool_outcomes["id1"]["status"] == "reconciled"
    assert "incomplete" not in restored.extensions
    with pytest.raises(RuntimeError, match="cannot be replayed"):
        run_tools_sync(tools, [], calls, memory=restored)


@pytest.mark.parametrize("cooperative", [False, True])
def test_external_executor_admission_reports_truthful_uncertainty(cooperative):
    async def run():
        abort, effects = asyncio.Event(), []

        class QueuedExecutor:
            manages_admission = cooperative

            async def aexecute(self, invocation):
                await asyncio.sleep(0)
                abort.set()
                invocation.check_cancelled()
                return invocation.execute()

        subject, model, _, _ = setup("model", [lambda: effects.append(0)])
        subject.executor = QueuedExecutor()
        with pytest.raises(asyncio.CancelledError):
            await drive(subject, "async", abort=abort)
        assert outcomes(model)[1] == ["not_started" if cooperative else "uncertain"]
        assert effects == []

    asyncio.run(run())


def test_single_completed_effect_with_failed_observer_requires_acknowledgement():
    from synaptic.agent.events import EventBus

    bus = EventBus(errors="raise")

    def fail(*args):
        raise LookupError("observer failed after receipt")

    bus.on("tool_end", fail)
    subject, model, _, _ = setup("agent", [lambda: "receipt"], events=bus)
    with pytest.raises(LookupError):
        subject.run("go")
    memory = model.history.entries[-1]
    assert len(memory.tool_results) == len(memory.tool_calls) == 1
    with pytest.raises(ValueError, match="Reconcile"):
        _runner.invoke(model, "continue")
    complete_tool_call(memory, memory.tool_results)
    assert memory.tool_outcomes["id0"]["status"] == "succeeded"
    _runner.invoke(model, "continue", autorun=False)


@pytest.mark.parametrize("boundary", ["permission", "tool_start", "step_end"])
def test_task_cancellation_before_admission_has_no_effect(boundary):
    from synaptic.agent.events import EventBus

    async def run():
        entered, effects, bus = asyncio.Event(), [], EventBus(errors="raise")

        async def wait(*args):
            entered.set()
            await asyncio.Future()

        options = {"permission": wait} if boundary == "permission" else {}
        if boundary != "permission":
            bus.on(boundary, wait)
        subject, model, _, _ = setup(
            "agent", [lambda: effects.append(0)], events=bus, **options
        )
        task = asyncio.create_task(drive(subject, "async"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert outcomes(model)[1] == ["not_started"] and effects == []

    asyncio.run(run())


def test_task_cancellation_during_agent_generation_retains_proposal():
    async def run():
        ready = asyncio.Event()
        subject, model, _, calls = setup("agent", [lambda: "never"])

        async def stream(**kwargs):
            yield ResponseChunk("", function_call=calls[0])
            ready.set()
            await asyncio.Future()

        model.llm.astream = stream
        task = asyncio.create_task(drive(subject, "async"))
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert outcomes(model)[1] == ["not_started"]
        assert subject.session.outcome == "cancelled"

    asyncio.run(run())


def test_legacy_external_receipt_cannot_be_replaced_after_reconciliation():
    memory = ResponseMem("", None, tool_calls=[ToolCall("external", {}, id="external-id")])
    complete_tool_call(memory, {"external-id": {"result": "receipt"}})
    with pytest.raises(ValueError, match="already recorded"):
        complete_tool_call(memory, [{"result": "changed"}])
    assert memory.tool_results == [{"result": "receipt"}]
