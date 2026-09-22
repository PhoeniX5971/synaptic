import asyncio
from types import SimpleNamespace

import pytest

from synaptic.mcp import DiscoveryLimits, MCPDiscoveryError, StdioMCPServer
from synaptic.mcp.session import MCPSession


def fake_session(list_tools, **limits):
    session = MCPSession(
        StdioMCPServer("test", "unused"), discovery_limits=DiscoveryLimits(**limits)
    )

    async def connect():
        pass

    session.connect = connect
    session.session = SimpleNamespace(list_tools=list_tools)
    return session


def page(name="tool", cursor=None):
    return SimpleNamespace(
        tools=[SimpleNamespace(name=name, inputSchema={"type": "object"})], nextCursor=cursor
    )


@pytest.mark.parametrize(
    "failure", ["repeat", "pages", "tools", "bytes", "timeout", "schema", "duplicate"]
)
def test_failed_refresh_preserves_old_catalog(failure):
    async def run():
        refreshing, calls = False, 0

        async def listing(cursor=None):
            nonlocal calls
            calls += 1
            if not refreshing:
                return page("old")
            if failure == "timeout":
                await asyncio.sleep(1)
            result = page("new" if failure == "duplicate" else str(calls), str(calls))
            if failure == "repeat":
                result.nextCursor = "repeated"
            if failure == "bytes":
                result.tools[0].name = "x" * 1000
            if failure == "schema":
                result.tools[0].inputSchema = {"type": "not-a-type"}
            return result

        limits = {"timeout": 0.05, "max_pages": 4, "max_tools": 4, "max_bytes": 700}
        if failure == "tools":
            limits["max_tools"] = 1
        session = fake_session(listing, **limits)
        snapshot = await session.list_tools()
        snapshot[0].name = "tampered"
        refreshing = True
        with pytest.raises(Exception):
            await session.list_tools(refresh=True)
        assert [tool.name for tool in await session.list_tools()] == ["old"]
        assert calls <= 5

    asyncio.run(run())


def test_concurrent_initial_discovery_is_coalesced():
    async def run():
        calls = 0

        async def listing(cursor=None):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return page()

        session = fake_session(listing)
        results = await asyncio.gather(*(session.list_tools() for _ in range(20)))
        assert calls == 1
        assert all(result[0].name == "tool" for result in results)

    asyncio.run(run())


@pytest.mark.parametrize(
    "limits", [{"max_pages": 0}, {"max_tools": -1}, {"max_bytes": True}, {"timeout": float("inf")}]
)
def test_invalid_limits_rejected(limits):
    with pytest.raises(ValueError):
        DiscoveryLimits(**limits)
