"""Credential-free real MCP server used only by local transport tests."""

import asyncio
import os
import sys

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

server = Server("synaptic-fixture")


@server.list_tools()
async def list_tools(request: types.ListToolsRequest) -> types.ListToolsResult:
    cursor = request.params.cursor if request and request.params else None
    name = "mixed" if cursor is None else "fail"
    return types.ListToolsResult(
        tools=[types.Tool(name=name, inputSchema={"type": "object", "properties": {}})],
        nextCursor="second" if cursor is None else None,
    )


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> types.CallToolResult:
    if name == "die":
        asyncio.get_running_loop().call_later(0.05, os._exit, 0)
    return types.CallToolResult(
        content=[
            types.TextContent(type="text", text="caption"),
            types.ImageContent(type="image", data="AA==", mimeType="image/png"),
        ],
        structuredContent={"pid": os.getpid()},
        isError=name == "fail",
        _meta={"fixture": True},
    )


_original_call = server.request_handlers[types.CallToolRequest]


async def protocol_error_handler(request):
    from mcp.shared.exceptions import McpError

    if request.params.name == "protocol":
        raise McpError(
            types.ErrorData(code=-32602, message="fixture protocol error", data={"field": "x"})
        )
    return await _original_call(request)


server.request_handlers[types.CallToolRequest] = protocol_error_handler


async def main():
    if len(sys.argv) > 1 and sys.argv[1] == "hang":
        await asyncio.Future()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
