import asyncio
import threading
from types import SimpleNamespace

import pytest

from synaptic import Agent, EventBus, History, Tool, ToolCall, ResponseChunk
from synaptic.agent.session import BudgetExceeded
from synaptic.core import _runner
from synaptic.core._ownership import operation
from synaptic.providers._bridge import stream
from synaptic.run import ThreadScheduler
from tests.test_completion_contract import fixture_model


def test_scheduler_is_responsive_and_cancelled_effect_keeps_capacity():
    async def run():
        scheduler = ThreadScheduler(1)
        gate, started = threading.Event(), threading.Event()
        effects = []

        def effect():
            started.set()
            gate.wait(2)
            effects.append(1)

        first = asyncio.create_task(scheduler.run(effect))
        while not started.is_set():
            await asyncio.sleep(0.001)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(scheduler.run(lambda: effects.append(2)))
        await asyncio.sleep(0.01)
        assert effects == [] and not second.done()
        gate.set()
        await asyncio.wait_for(second, 1)
        assert effects == [1, 2]
        scheduler.close()
        with pytest.raises(RuntimeError):
            await scheduler.run(lambda: None)
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(anext(stream(lambda: iter([]), scheduler)), 1)

    asyncio.run(run())


def test_model_and_shared_history_reject_concurrent_owners():
    async def run():
        history = History()
        one, two = SimpleNamespace(history=history), SimpleNamespace(history=history)

        async def conflicting():
            with operation(two):
                pass

        with operation(one):
            with operation(one):
                pass
            with pytest.raises(RuntimeError):
                await asyncio.create_task(conflicting())
        with operation(two):
            pass

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["error", "cancel", "close", "budget", "success"])
def test_agent_stream_outcomes_and_source_cleanup(mode):
    async def run():
        closed = []
        model = fixture_model([ResponseChunk("", is_final=True)])

        async def source(**kwargs):
            try:
                yield ResponseChunk("partial")
                if mode == "error":
                    raise ValueError("provider failure")
                if mode == "cancel":
                    raise asyncio.CancelledError()
                if mode == "budget":
                    yield ResponseChunk("", function_call=ToolCall("echo", {}))
                yield ResponseChunk("partial", is_final=True)
            finally:
                closed.append(True)

        model.llm.astream = source
        tool = Tool("echo", {"parameters": {"type": "object"}}, lambda: "ok", add_to_registry=False)
        model.blacklist = []
        agent = Agent(model, tools=[tool], max_turns=1)
        output = agent.astream("prompt")
        if mode == "close":
            assert (await anext(output)).text == "partial"
            await output.aclose()
        elif mode in ("error", "cancel", "budget"):
            error = {
                "error": ValueError,
                "cancel": asyncio.CancelledError,
                "budget": BudgetExceeded,
            }[mode]
            with pytest.raises(error):
                _ = [item async for item in output]
        else:
            _ = [item async for item in output]
        assert closed == [True]
        assert (
            agent.session.outcome
            == {
                "close": "cancelled",
                "cancel": "cancelled",
                "error": "failed",
                "budget": "exhausted",
                "success": "completed",
            }[mode]
        )
        if mode in ("error", "cancel", "close"):
            assert agent.session.history.entries[-1].extensions["incomplete"] is True
        assert getattr(model, "_operation_owner", None) is None

    asyncio.run(run())


def test_stream_authorizes_effective_defaults_and_external_executor():
    async def run():
        checked, executed = [], []
        call = ToolCall("write", {})
        tool = Tool(
            "write",
            {"parameters": {"type": "object"}},
            lambda path: executed.append(path),
            default_params={"path": "approved"},
            add_to_registry=False,
        )
        model = fixture_model(
            [ResponseChunk("", function_call=call), ResponseChunk("", is_final=True)]
        )

        def policy(name, args):
            checked.append(dict(args))
            return args["path"] == "approved"

        class External:
            async def aexecute(self, invocation):
                assert dict(invocation.arguments) == checked[-1]
                return invocation.execute()

        model.blacklist = []
        agent = Agent(model, tools=[tool], permission=policy, executor=External(), max_turns=1)
        with pytest.raises(BudgetExceeded):
            _ = [item async for item in agent.astream("prompt")]
        assert checked == [{"path": "approved"}] and executed == ["approved"]
        assert agent.session.history.entries[-1].tool_results[0]["call_id"] == call.id

    asyncio.run(run())


def test_observer_failure_and_disposal_are_independent_of_result():
    failures, seen = [], []
    events = EventBus(on_error=lambda *args: failures.append(args))
    subscription = events.on("x", lambda value: (_ for _ in ()).throw(ValueError("failed")))
    events.on("x", lambda value: seen.append(value))
    events.emit("x", {"ok": 1})
    subscription.close()
    events.emit("x", {"ok": 2})
    assert len(failures) == 1 and seen == [{"ok": 1}, {"ok": 2}]
