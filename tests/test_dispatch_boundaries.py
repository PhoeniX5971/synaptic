"""Cancellation and receipt retention at every dispatch ownership boundary."""

import asyncio
import json
import threading

import pytest

from synaptic import Agent, ToolCall, History
from synaptic.agent.events import EventBus
from synaptic.core import _runner
from synaptic.core.invocation import LocalExecutor
from synaptic.core._tool_execution import run_tools_sync
from synaptic.run.scheduling import ThreadScheduler
from tests.test_execution_recovery import make_model, make_tool


async def drive(subject, mode, **kwargs):
    if isinstance(subject, Agent):
        if mode == "sync":
            return subject.run("go", **kwargs)
        if mode == "async":
            return await subject.arun("go", **kwargs)
        return [chunk async for chunk in subject.astream("go", **kwargs)]
    kwargs.update(autorun=True, automem=True)
    if mode == "sync":
        return _runner.invoke(subject, "go", **kwargs)
    if mode == "async":
        return await _runner.ainvoke(subject, "go", **kwargs)
    return [chunk async for chunk in _runner.astream(subject, "go", **kwargs)]


def setup(kind, functions, **kwargs):
    tools = [make_tool(f"tool{i}", fn) for i, fn in enumerate(functions)]
    calls = [ToolCall(tool.name, {}, id=f"id{i}") for i, tool in enumerate(tools)]
    model = make_model(tools, calls)
    subject = Agent(model, tools=tools, max_turns=1, **kwargs) if kind == "agent" else model
    return subject, model, tools, calls


def outcomes(model):
    memory = History.from_dict(json.loads(json.dumps(model.history.to_dict()))).entries[-1]
    return memory, [item["status"] for item in memory.tool_outcomes.values()]


@pytest.mark.parametrize("kind", ["model", "agent"])
@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
@pytest.mark.parametrize("external", [False, True])
def test_second_failure_preserves_receipt_and_blocks_replay(kind, mode, external):
    async def run():
        effects = []

        def first():
            effects.append(0)
            return "receipt"

        def second():
            effects.append(1)
            raise ValueError("effect may have happened")

        subject, model, tools, calls = setup(kind, [first, second, lambda: effects.append(2)])
        if external:

            class External:
                def execute(self, invocation):
                    return invocation.execute()

                async def aexecute(self, invocation):
                    return invocation.execute()

            subject.executor = External()
        with pytest.raises(ValueError):
            await drive(subject, mode)
        memory, statuses = outcomes(model)
        assert statuses == ["succeeded", "uncertain", "not_started"]
        assert memory.tool_results == [{"name": "tool0", "call_id": "id0", "result": "receipt"}]
        with pytest.raises(ValueError, match="Reconcile"):
            await drive(model, "sync")
        with pytest.raises(RuntimeError, match="cannot be replayed"):
            run_tools_sync(tools, [], calls, memory=memory)
        assert effects == [0, 1]

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
@pytest.mark.parametrize(
    "event, statuses",
    [
        ("step_end", ["not_started"] * 2),
        ("tool_start", ["not_started"] * 2),
        ("tool_end", ["succeeded", "not_started"]),
    ],
)
def test_observer_failure_retains_receipt(mode, event, statuses):
    async def run():
        effects, bus = [], EventBus(errors="raise")

        def fail(*args):
            raise LookupError("observer failed")

        bus.on(event, fail)
        subject, model, _, _ = setup(
            "agent", [lambda: effects.append(0), lambda: effects.append(1)], events=bus
        )
        with pytest.raises(LookupError):
            await drive(subject, mode)
        memory, actual = outcomes(model)
        assert actual == statuses
        assert len(memory.tool_results) == statuses.count("succeeded")
        assert memory.extensions["incomplete"]

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["async", "stream"])
@pytest.mark.parametrize("boundary", ["permission", "step_end", "tool_start", "tool_end"])
def test_abort_after_awaited_policy_or_observer(mode, boundary):
    async def run():
        abort, effects, bus = asyncio.Event(), [], EventBus(errors="raise")

        async def cancel(*args):
            await asyncio.sleep(0)
            abort.set()
            return True

        kwargs = {"permission": cancel} if boundary == "permission" else {}
        if boundary != "permission":
            bus.on(boundary, cancel)
        subject, model, _, _ = setup(
            "agent",
            [lambda: effects.append(0), lambda: effects.append(1)],
            events=bus,
            **kwargs,
        )
        with pytest.raises(asyncio.CancelledError):
            await drive(subject, mode, abort=abort)
        _, statuses = outcomes(model)
        assert statuses == (
            ["succeeded", "not_started"] if boundary == "tool_end" else ["not_started"] * 2
        )
        assert effects == ([0] if boundary == "tool_end" else [])
        assert subject.session.outcome == "cancelled"

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["model", "agent"])
@pytest.mark.parametrize("mode", ["async", "stream"])
def test_task_cancel_during_second_effect_is_uncertain(kind, mode):
    async def run():
        started = asyncio.Event()

        async def second():
            started.set()
            await asyncio.Future()

        subject, model, _, _ = setup(kind, [lambda: "receipt", second, lambda: "never"])
        task = asyncio.create_task(drive(subject, mode))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        memory, statuses = outcomes(model)
        assert statuses == ["succeeded", "uncertain", "not_started"]
        assert memory.tool_results[0]["result"] == "receipt"

    asyncio.run(run())


@pytest.mark.parametrize("cancel_started", [False, True])
def test_thread_admission_and_inflight_cancellation(cancel_started):
    async def run():
        scheduler = ThreadScheduler(workers=1)
        release, started, effects, abort = (
            threading.Event(),
            threading.Event(),
            [],
            asyncio.Event(),
        )

        def occupy():
            started.set()
            release.wait(5)
            return "receipt"

        subject, model, _, _ = setup(
            "model", [occupy if cancel_started else lambda: effects.append(0)]
        )
        subject.executor = LocalExecutor(scheduler)
        blocker = None
        if not cancel_started:
            blocker = asyncio.create_task(scheduler.run(occupy))
        task = asyncio.create_task(drive(subject, "async", abort=abort))
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            if cancel_started:
                task.cancel()
            else:
                abort.set()
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            _, statuses = outcomes(model)
            assert statuses == ["uncertain" if cancel_started else "not_started"]
            assert effects == []
        finally:
            release.set()
            if blocker:
                await blocker
            scheduler.close()

    asyncio.run(run())


def test_task_cancel_while_waiting_for_scheduler_admission():
    async def run():
        scheduler = ThreadScheduler(workers=1)
        release, started, effects = threading.Event(), threading.Event(), []

        def occupy():
            started.set()
            release.wait(5)

        blocker = asyncio.create_task(scheduler.run(occupy))
        while not started.is_set():
            await asyncio.sleep(0.001)
        subject, model, _, _ = setup("model", [lambda: effects.append(0)])
        subject.executor = LocalExecutor(scheduler)
        task = asyncio.create_task(drive(subject, "async"))
        try:
            # Let generation finish and reach the occupied scheduler.
            while getattr(model, "last_response", None) is None:
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert outcomes(model)[1] == ["not_started"]
            release.set()
            await blocker
            await scheduler.run(lambda: None)
            assert effects == []
        finally:
            release.set()
            scheduler.close()

    asyncio.run(run())
