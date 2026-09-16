import asyncio
from types import SimpleNamespace

import pytest

from synaptic.mcp import MCPHub, StdioMCPServer
from synaptic.mcp.tools import tool_name
from tests.test_mcp import FakeSession


def test_aliases_are_bounded_collision_safe_and_reversible():
    names = [tool_name("a", "b__c"), tool_name("a__b", "c"), tool_name("x" * 100, "y" * 100)]
    assert len(set(names)) == 3
    assert all(len(name) <= 64 for name in names)

    async def run():
        hub = MCPHub([StdioMCPServer("a__b", "unused")])
        hub.sessions["a__b"] = FakeSession()
        alias = (await hub.load_server("a__b"))[0]
        assert hub.resolve_alias(alias).server == "a__b"
        await hub.close()

    asyncio.run(run())


def test_failed_preparation_preserves_catalog_and_consistent_observer_snapshot():
    async def run():
        hub = MCPHub([StdioMCPServer("test", "unused")])
        fake = FakeSession()
        hub.sessions["test"] = fake
        await hub.load_server("test")
        old = dict(hub.registry.all())
        seen = []
        hub.registry.on_change(
            lambda: seen.append(set(hub.loaded["test"]) <= set(hub.registry.all()))
        )
        await hub.load_server("test", refresh=True)
        assert seen == [True]

        async def invalid(refresh=False):
            return [SimpleNamespace(name="bad", inputSchema={"type": "invalid"})]

        fake.list_tools = invalid
        with pytest.raises(Exception):
            await hub.load_server("test", refresh=True)
        assert set(hub.registry.all()) == set(old)

    asyncio.run(run())


def test_bounded_workers_failure_isolation_and_draining():
    async def run():
        hub = MCPHub([StdioMCPServer(str(i), "unused") for i in range(8)], concurrency=2)
        active = peak = 0

        class Slow(FakeSession):
            async def list_tools(self, refresh=False):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1
                return await super().list_tools(refresh)

        hub.sessions = {name: Slow() for name in hub.sessions}

        async def fail(refresh=False):
            raise ConnectionError("unavailable")

        hub.sessions["0"].list_tools = fail
        results = await hub.load_all()
        assert isinstance(results["0"], ConnectionError)
        assert len(hub.loaded) == 7 and peak <= 2
        gate, started = asyncio.Event(), asyncio.Event()

        async def slow_call(*args, **kwargs):
            started.set()
            await gate.wait()
            return SimpleNamespace(content=[])

        hub.sessions["1"].call_tool = slow_call
        tool = hub.registry.all()[hub.loaded["1"][0]]
        call = asyncio.create_task(tool.run(text="x"))
        await started.wait()
        removal = asyncio.create_task(hub.remove_server("1"))
        await asyncio.sleep(0.01)
        assert not removal.done()
        gate.set()
        await call
        await removal
        with pytest.raises(RuntimeError):
            await tool.run(text="x")
        await hub.close()

    asyncio.run(run())


def test_connection_capacity_eviction_backoff_and_all_close_attempts():
    async def run():
        hub = MCPHub([StdioMCPServer(name, "unused") for name in ("a", "b")], max_connections=1)
        hub.sessions = {name: FakeSession() for name in hub.sessions}
        await hub.load_server("a")
        await hub.load_server("b")
        assert hub.sessions["a"].closed and hub.runtime.connected == {"b"}
        closed = []

        async def fail():
            closed.append("a")
            raise RuntimeError("close failure")

        async def good():
            closed.append("b")

        hub.sessions["a"].close, hub.sessions["b"].close = fail, good
        with pytest.raises(ExceptionGroup):
            await hub.close()
        assert set(closed) == {"a", "b"} and not hub.loaded and not hub.registry.all()

    asyncio.run(run())


def test_independent_hubs_cannot_rebind_owned_aliases_and_routing_closes():
    async def run():
        from synaptic import ToolRegistry

        registry = ToolRegistry()
        one = MCPHub([StdioMCPServer("same", "unused")], registry)
        two = MCPHub([StdioMCPServer("same", "unused")], registry)
        one.sessions["same"] = FakeSession()
        two.sessions["same"] = FakeSession()
        await one.load_server("same")
        original = dict(registry.all())
        with pytest.raises(ValueError):
            await two.load_server("same")
        assert dict(registry.all()) == original
        await two.close()
        assert dict(registry.all()) == original
        one.enable_routing()
        await one.close()
        assert not registry.all()

    asyncio.run(run())
