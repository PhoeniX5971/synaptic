"""Real local transports; no credentials or external network endpoints."""

import asyncio
import os
from pathlib import Path
import socket
import sys

import pytest

pytest.importorskip("mcp")

from synaptic.mcp import ConnectionState, HttpMCPServer, StdioMCPServer, normalize_result
from synaptic.mcp.session import MCPSession
from synaptic.mcp import MCPProtocolError

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_server.py"


def stdio(*args):
    return MCPSession(
        StdioMCPServer("fixture", sys.executable, [str(FIXTURE), *args]),
        connect_timeout=3,
        shutdown_timeout=3,
        health_interval=0.05,
        request_timeout=0.3,
    )


def test_real_stdio_pagination_mixed_results_close_reconnect_and_eof():
    async def run():
        session = stdio()
        try:
            await asyncio.gather(session.connect(), session.connect())
            assert [tool.name for tool in await session.list_tools()] == ["mixed", "fail"]
            result = normalize_result(await session.call_tool("mixed", {}))
            pid = result["structured_content"]["pid"]
            assert result["content"][1]["type"] == "image"
            assert result["metadata"] == {"fixture": True}
            assert normalize_result(await session.call_tool("fail", {}))["is_error"]
            with pytest.raises(MCPProtocolError) as caught:
                await session.call_tool("protocol", {})
            assert caught.value.code == -32602
            assert caught.value.data == {"field": "x"}
            assert session.protocol_version == "2025-11-25"
            await session.close()
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            await session.reconnect()
            await session.call_tool("die", {})
            async with asyncio.timeout(2):
                while session.state == ConnectionState.CONNECTED:
                    await asyncio.sleep(0.01)
            assert session.session is None
            with pytest.raises(Exception):
                await session.connect()
        finally:
            await session.close()
            await session.close()

    asyncio.run(run())


def test_real_stdio_missing_executable_and_cancel_initialization():
    async def run():
        session = MCPSession(StdioMCPServer("missing", "/nonexistent/synaptic-fixture"))
        try:
            outcomes = await asyncio.wait_for(
                asyncio.gather(session.connect(), session.connect(), return_exceptions=True), 3
            )
            assert all(isinstance(outcome, Exception) for outcome in outcomes)
        finally:
            await session.close()
        session = stdio("hang")
        task = asyncio.create_task(session.connect())
        await asyncio.sleep(0.1)
        await session.close()
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(task, 1)
        assert session._task.done()

    asyncio.run(run())


def test_real_loopback_http_headers_and_result_round_trip():
    from mcp.server.fastmcp import FastMCP
    import uvicorn

    async def run():
        server = FastMCP("http-fixture", stateless_http=True, json_response=True)

        @server.tool()
        async def echo(text: str) -> dict[str, str]:
            return {"echo": text}

        app = server.streamable_http_app()
        seen = []

        async def checked_app(scope, receive, send):
            if scope["type"] == "http":
                seen.append(dict(scope["headers"]).get(b"x-fixture"))
            await app(scope, receive, send)

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        runner = uvicorn.Server(uvicorn.Config(checked_app, log_level="error", lifespan="on"))
        task = asyncio.create_task(runner.serve(sockets=[sock]))
        session = MCPSession(
            HttpMCPServer("http", f"http://127.0.0.1:{port}/mcp", headers={"x-fixture": "present"})
        )
        try:
            async with asyncio.timeout(3):
                while not runner.started:
                    await asyncio.sleep(0.01)
            assert [tool.name for tool in await session.list_tools()] == ["echo"]
            result = normalize_result(await session.call_tool("echo", {"text": "hello"}))
            assert result["structured_content"] == {"echo": "hello"}
            assert seen and all(header == b"present" for header in seen)
        finally:
            await session.close()
            runner.should_exit = True
            await asyncio.wait_for(task, 3)
            sock.close()

    asyncio.run(run())
