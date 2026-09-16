import asyncio
from types import SimpleNamespace

import pytest

from synaptic.mcp import StdioMCPServer, ConnectionState
from synaptic.mcp.session import MCPSession


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_late_dispatch_preserves_replacement_with_concurrent_current_calls(outcome):
    async def run():
        session = MCPSession(StdioMCPServer("fixture", "unused"), request_timeout=0.1)
        gate, started = asyncio.Event(), asyncio.Event()
        owners = []

        async def owner():
            generation = len(owners)
            owners.append(asyncio.current_task())

            async def call(name, arguments):
                if generation == 0:
                    started.set()
                    await gate.wait()
                    if outcome == "failure":
                        raise ConnectionError("old failed")
                return generation

            session.session = SimpleNamespace(call_tool=call)
            session.state = ConnectionState.CONNECTED
            session._ready.set()
            await asyncio.Future()

        session._run_stdio = owner
        await session.connect()
        old = asyncio.create_task(session.call_tool("echo", {}))
        await started.wait()
        await session.reconnect()
        replacement = session.session
        assert await asyncio.gather(
            session.call_tool("echo", {}), session.call_tool("echo", {})
        ) == [1, 1]
        if outcome != "timeout":
            gate.set()
        if outcome == "success":
            assert await old == 0
        else:
            with pytest.raises((ConnectionError, TimeoutError)):
                await old
        assert session.state == ConnectionState.CONNECTED and session.session is replacement
        assert not owners[1].done() and await session.call_tool("echo", {}) == 1
        await session.close()
        assert all(task.done() for task in owners)

    asyncio.run(run())


def test_failure_invalidates_only_current_owner():
    async def run():
        session = MCPSession(StdioMCPServer("fixture", "unused"))

        async def fail(*args, **kwargs):
            raise ConnectionError("current failed")

        async def owner():
            session.session = SimpleNamespace(call_tool=fail)
            session.state = ConnectionState.CONNECTED
            session._ready.set()
            await asyncio.Future()

        session._run_stdio = owner
        await session.connect()
        with pytest.raises(ConnectionError):
            await session.call_tool("echo", {})
        assert session.session is None and session.state == ConnectionState.FAILED
        await session.close()
        assert session._task is None or session._task.done()

    asyncio.run(run())
