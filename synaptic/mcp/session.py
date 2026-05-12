from contextlib import AsyncExitStack
from typing import Any, List, Optional

from .servers import HttpMCPServer, MCPServer, StdioMCPServer


class MCPSession:
    def __init__(self, server: MCPServer) -> None:
        self.server = server
        self.session: Any = None
        self._stack: Optional[AsyncExitStack] = None
        self._tools: Optional[List[Any]] = None

    async def connect(self) -> None:
        if self.session is not None:
            return
        self._stack = AsyncExitStack()
        if isinstance(self.server, StdioMCPServer):
            await self._connect_stdio()
        elif isinstance(self.server, HttpMCPServer):
            await self._connect_http()
        else:
            raise TypeError("Unsupported MCP server config")
        await self.session.initialize()

    async def list_tools(self, refresh: bool = False) -> List[Any]:
        await self.connect()
        if self._tools is None or refresh:
            result = await self.session.list_tools()
            self._tools = list(getattr(result, "tools", []))
        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> Any:
        await self.connect()
        return await self.session.call_tool(name, arguments=arguments)

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self.session = None
        self._stack = None
        self._tools = None

    async def _connect_stdio(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.server.command,
            args=self.server.args,
            env=self.server.env,
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))

    async def _connect_http(self) -> None:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        read, write, _ = await self._stack.enter_async_context(
            streamable_http_client(self.server.url, headers=self.server.headers)
        )
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
