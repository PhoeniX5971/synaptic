"""SDK transport contexts: entered and exited exclusively by the owner task."""

from contextlib import asynccontextmanager

from .servers import HttpMCPServer, StdioMCPServer


@asynccontextmanager
async def open_transport(server):
    try:
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:
        raise ImportError("Install Synaptic's MCP extra: pip install 'synaptic[mcp]'") from exc
    if isinstance(server, StdioMCPServer):
        params = StdioServerParameters(command=server.command, args=server.args, env=server.env)
        async with stdio_client(params) as streams:
            yield streams
    elif isinstance(server, HttpMCPServer):
        import httpx

        async with httpx.AsyncClient(headers=server.headers, timeout=30.0) as client:
            async with streamable_http_client(server.url, http_client=client) as (read, write, _):
                yield read, write
    else:
        raise TypeError("Unsupported MCP server config")
