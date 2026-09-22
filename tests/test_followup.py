"""Follow-up review regressions; recorded failing before fixes."""

import asyncio
from types import SimpleNamespace

import pytest

from synaptic import Agent, EventBus, Tool, ToolCall, ToolRegistry
from synaptic.mcp import ConnectionState, MCPHub, StdioMCPServer
from synaptic.mcp.session import MCPSession
from tests.test_mcp import FakeSession


def test_old_request_failure_cannot_destroy_replacement():
    async def run():
        session = MCPSession(StdioMCPServer("fixture", "unused"))
        gate, started = asyncio.Event(), asyncio.Event()

        async def old_call(*args, **kwargs):
            started.set()
            await gate.wait()
            raise ConnectionError("old request failed")

        async def owner():
            session.session = SimpleNamespace(call_tool=old_call)
            session.state = ConnectionState.CONNECTED
            session._ready.set()
            await asyncio.Future()

        session._run_stdio = owner
        await session.connect()
        request = asyncio.create_task(session.call_tool("echo", {}))
        await started.wait()
        await session.reconnect()
        replacement = session.session
        gate.set()
        with pytest.raises(ConnectionError):
            await request
        assert session.state == ConnectionState.CONNECTED
        assert session.session is replacement
        await session.close()

    asyncio.run(run())


def hub_fixture():
    hub = MCPHub([StdioMCPServer("a", "unused"), StdioMCPServer("a__b", "unused")])
    hub.sessions = {name: FakeSession() for name in hub.sessions}
    return hub


def test_unload_preserves_overlapping_server_ownership():
    async def run():
        hub = hub_fixture()
        await hub.load_all()
        names = list(hub.loaded["a__b"])
        hub.unload_server("a")
        assert all(name in hub.registry.all() for name in names)
        await hub.close()

    asyncio.run(run())


def test_observer_failure_does_not_destroy_published_catalog():
    async def run():
        hub = hub_fixture()
        await hub.load_server("a")

        def fail():
            raise RuntimeError("observer failed")

        hub.registry.on_change(fail)
        try:
            await hub.load_server("a", refresh=True)
        except RuntimeError:
            pass
        assert hub.loaded.get("a")
        assert set(hub.loaded["a"]) <= set(hub.registry.all())

    asyncio.run(run())


def test_observer_failure_cannot_prevent_session_cleanup():
    async def run():
        hub = hub_fixture()
        await hub.load_all()

        def fail():
            raise RuntimeError("observer failed")

        hub.registry.on_change(fail)
        try:
            await hub.close()
        except RuntimeError:
            pass
        assert all(session.closed for session in hub.sessions.values())

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_policy_sees_effective_defaults_before_execution(mode):
    effects, checked = [], []
    tool = Tool(
        "write",
        {
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }
        },
        lambda path: effects.append(path),
        default_params={"path": "forbidden"},
        add_to_registry=False,
    )
    agent = Agent.__new__(Agent)
    agent.events, agent.signals = EventBus(), None
    agent._registry = ToolRegistry()
    agent._registry.register(tool)
    agent.model = SimpleNamespace(blacklist=[])
    agent.session = SimpleNamespace(id="fixture")

    def allow(name, args):
        checked.append(dict(args))
        return args.get("path") != "forbidden"

    agent.permission = allow
    calls = [ToolCall("write", {})]
    if mode == "sync":
        agent._tool_results(calls)
    else:
        asyncio.run(agent._atool_results(calls))
    assert checked == [{"path": "forbidden"}]
    assert effects == []
