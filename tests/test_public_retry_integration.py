import asyncio
import json
from types import SimpleNamespace
import pytest
from synaptic import History, ResponseMem, ResponseChunk, Tool, ToolCall
from synaptic.run import Retry
from tests.test_public_integration import model_with


@pytest.mark.parametrize("mode", ["async", "stream"])
@pytest.mark.parametrize(
    "failure", ["repeated", "deadline", "cancel", "output", "unresolved", "effect"]
)
def test_retry_barriers_and_retained_attempt_evidence(mode, failure):
    async def run():
        calls, effects = [], []

        async def source(**kwargs):
            calls.append(1)
            if failure in ("deadline", "cancel"):
                if failure == "cancel":
                    asyncio.current_task().cancel()
                await asyncio.sleep(5)
            if failure == "output":
                yield ResponseChunk("visible")
            if failure == "effect":
                yield ResponseChunk("", function_call=ToolCall("effect", {}, id="effect-id"))
                yield ResponseChunk("", is_final=True)
                return
            raise ConnectionError("transient")

        def effect():
            effects.append(1)
            raise ConnectionError("effect started")

        tool = Tool("effect", {"parameters": {"type": "object"}}, effect, add_to_registry=False)
        model = model_with(
            SimpleNamespace(astream=source, synaptic_tools=[tool]),
            automem=True,
            autorun=failure == "effect",
        )
        if failure == "unresolved":
            model.history.add(ResponseMem("old", None, extensions={"incomplete": True}))
        retry = Retry(
            model,
            max_retries=2,
            delay=0,
            jitter=0,
            deadline=0.03 if failure == "deadline" else 10,
        )
        expected = {
            "cancel": asyncio.CancelledError,
            "deadline": TimeoutError,
            "unresolved": ValueError,
        }.get(failure, ConnectionError)
        with pytest.raises(expected):
            if mode == "async":
                await retry.ainvoke("go")
            else:
                _ = [chunk async for chunk in retry.astream("go")]
        assert len(calls) == (
            3 if failure == "repeated" else 0 if failure == "unresolved" else 1
        )
        restored = History.from_dict(json.loads(json.dumps(model.history.to_dict())))
        if failure in ("repeated", "deadline", "cancel"):
            assert len(restored.failed_attempts) == len(calls)
            assert restored.entries == []
            assert all(m.extensions["incomplete"] for m in restored.failed_attempts)
        if failure == "effect":
            assert effects == [1]
            assert restored.recovery[-1].tool_outcomes["effect-id"]["status"] == "uncertain"
        if failure == "output":
            assert restored.entries[-1].message == "visible"
            assert restored.entries[-1].extensions["incomplete"]

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["async", "stream"])
def test_autorun_failure_before_output_keeps_normal_recovery_barrier(mode):
    async def run():
        calls = []

        async def source(**kwargs):
            calls.append(1)
            raise ConnectionError("no output")
            yield

        model = model_with(SimpleNamespace(astream=source), automem=True, autorun=True)
        with pytest.raises(ConnectionError):
            wrapped = Retry(model, max_retries=2, delay=0)
            if mode == "async":
                await wrapped.ainvoke("go")
            else:
                _ = [chunk async for chunk in wrapped.astream("go")]
        assert calls == [1]
        assert model.history.entries[-1].extensions["incomplete"]
        assert model.history.failed_attempts == []

    asyncio.run(run())


def test_model_retry_backoff_cancellation_preserves_attempt_and_releases_owner():
    async def run():
        entered = asyncio.Event()
        calls = []

        async def source(**kwargs):
            calls.append(1)
            entered.set()
            raise ConnectionError("transient")
            yield

        model = model_with(SimpleNamespace(astream=source), automem=True)
        task = asyncio.create_task(Retry(model, delay=5, jitter=0).ainvoke("go"))
        await entered.wait()
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="Concurrent"):
            await model.ainvoke("other")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == [1]
        assert model.history.entries == [] and len(model.history.failed_attempts) == 1
        assert getattr(model.history, "_operation_owner", None) is None

    asyncio.run(run())
