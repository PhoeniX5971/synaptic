import asyncio
import gc
import warnings

import pytest

from synaptic.agent import _perm
from synaptic.agent.events import EventBus
from synaptic.core._runner import run_tools_async, run_tools_sync
from synaptic.core.tool import Tool, ToolCall
from synaptic.run import Flow, Pipeline


class CustomAwaitable:
    def __await__(self):
        async def value():
            return 42

        return value().__await__()


@pytest.mark.parametrize("kind", ["coroutine", "future", "custom"])
def test_async_boundaries_resolve_general_awaitables(kind):
    async def run():
        def make(*args):
            if kind == "custom":
                return CustomAwaitable()
            if kind == "future":
                result = asyncio.get_running_loop().create_future()
                result.set_result(42)
                return result

            async def value():
                return 42

            return value()

        tool = Tool("awaitable", {}, make, add_to_registry=False)
        assert await run_tools_async([tool], [], [ToolCall(tool.name, {})]) == [
            {"name": tool.name, "result": 42}
        ]
        flow = Flow()
        flow.add("value", make)
        flow.entry("value")
        assert await flow.arun(None) == 42
        assert await Pipeline([make]).arun(None) == 42
        assert await _perm.aallow(make, ToolCall("value", {}))
        bus = EventBus()
        bus.on("event", make)
        await bus.aemit("event")

    asyncio.run(run())


def test_sync_rejection_closes_coroutines():
    async def value(*args):
        return 42

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flow = Flow()
        flow.add("value", value)
        flow.entry("value")
        for fn in [
            lambda: flow.run(None),
            lambda: Pipeline([value]).run(None),
            lambda: _perm.allow(value, ToolCall("value", {})),
        ]:
            with pytest.raises(RuntimeError):
                fn()
        bus = EventBus()
        bus.on("event", value)
        with pytest.raises(RuntimeError):
            bus.emit("event")
        tool = Tool("value", {}, lambda: value(), add_to_registry=False)
        assert "error" in run_tools_sync([tool], [], [ToolCall("value", {})])[0]
        gc.collect()
        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_sync_rejection_does_not_cancel_caller_owned_future():
    async def run():
        future = asyncio.get_running_loop().create_future()
        with pytest.raises(RuntimeError):
            Pipeline([lambda _: future]).run(None)
        assert not future.cancelled()
        future.cancel()

    asyncio.run(run())
