"""Executable reproductions of reviewed MCP defects (BUG-001 through BUG-003)."""

import asyncio
from types import SimpleNamespace

import pytest

from synaptic.mcp.results import normalize_result
from synaptic.mcp.servers import StdioMCPServer
from synaptic.mcp.session import MCPSession


def test_normalization_keeps_mixed_content():
    result = normalize_result(
        SimpleNamespace(
            structuredContent={"count": 1},
            isError=False,
            _meta={"trace": "abc"},
            content=[
                SimpleNamespace(type="text", text="caption"),
                SimpleNamespace(type="image", data="AA==", mimeType="image/png"),
            ],
        )
    )
    assert result["structured_content"] == {"count": 1}
    assert result["content"][1]["mimeType"] == "image/png"
    assert result["metadata"] == {"trace": "abc"}


def test_discovery_collects_every_page():
    async def run():
        session = MCPSession(StdioMCPServer("test", "unused"))

        async def connect():
            pass

        async def list_tools(cursor=None):
            return SimpleNamespace(
                tools=[SimpleNamespace(name=cursor or "first", inputSchema={})],
                nextCursor="second" if cursor is None else None,
            )

        session.connect = connect
        session.session = SimpleNamespace(list_tools=list_tools)
        assert [t.name for t in await session.list_tools()] == ["first", "second"]

    asyncio.run(run())


def test_all_connection_waiters_receive_startup_error():
    async def run():
        session = MCPSession(StdioMCPServer("test", "unused"))

        async def fail():
            await asyncio.sleep(0)
            session._error = RuntimeError("startup failed")
            session._ready.set()

        session._run_stdio = fail
        results = await asyncio.wait_for(
            asyncio.gather(session.connect(), session.connect(), return_exceptions=True), 1
        )
        assert all(isinstance(result, RuntimeError) for result in results)
        await session.close()

    asyncio.run(run())
