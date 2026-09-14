import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from synaptic.mcp import MCPProtocolError, MCPTransportError, StdioMCPServer
from synaptic.mcp.session import MCPSession


@pytest.mark.parametrize("kind", ["protocol", "transport", "tool"])
def test_call_failure_categories_and_protocol_details(kind):
    async def run():
        session = MCPSession(StdioMCPServer("fake", "unused"))

        async def connect():
            pass

        async def call_tool(*args, **kwargs):
            if kind == "protocol":
                raise McpError(
                    ErrorData(code=-32602, message="invalid params", data={"field": "x"})
                )
            if kind == "transport":
                raise ConnectionError("closed")
            return SimpleNamespace(isError=True, content=[])

        session.connect = connect
        session.session = SimpleNamespace(call_tool=call_tool)
        if kind == "tool":
            assert (await session.call_tool("echo", {})).isError
        else:
            error_type = MCPProtocolError if kind == "protocol" else MCPTransportError
            with pytest.raises(error_type) as caught:
                await session.call_tool("echo", {})
            assert caught.value.__cause__ is not None
            if kind == "protocol":
                assert caught.value.code == -32602
                assert caught.value.data == {"field": "x"}

    asyncio.run(run())
