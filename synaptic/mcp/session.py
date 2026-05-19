import asyncio
import contextlib
from typing import Any, List, Optional

from .servers import HttpMCPServer, MCPServer, StdioMCPServer


class MCPSession:
    def __init__(self, server: MCPServer) -> None:
        self.server = server
        self.session: Any = None
        self._tools: Optional[List[Any]] = None
        self._task: Optional[asyncio.Task] = None
        self._ready: Optional[asyncio.Event] = None
        self._error: Optional[BaseException] = None

    async def connect(self) -> None:
        if self.session is not None:
            return
        self._ready = asyncio.Event()
        self._error = None
        if isinstance(self.server, StdioMCPServer):
            self._task = asyncio.create_task(self._run_stdio())
        elif isinstance(self.server, HttpMCPServer):
            self._task = asyncio.create_task(self._run_http())
        else:
            raise TypeError("Unsupported MCP server config")
        await self._ready.wait()
        if self._error is not None:
            raise self._error

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
        task, self._task = self._task, None
        self._tools = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(Exception):
            await task

    async def _run_stdio(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        assert self._ready is not None
        assert isinstance(self.server, StdioMCPServer)
        params = StdioServerParameters(
            command=self.server.command,
            args=self.server.args,
            env=self.server.env,
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()
                    self._ready.set()
                    await asyncio.get_running_loop().create_future()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._error = exc
            if not self._ready.is_set():
                self._ready.set()
        finally:
            self.session = None

    async def _run_http(self) -> None:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        assert self._ready is not None
        assert isinstance(self.server, HttpMCPServer)
        try:
            async with streamable_http_client(self.server.url, headers=self.server.headers) as (read, write, _):
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()
                    self._ready.set()
                    await asyncio.get_running_loop().create_future()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._error = exc
            if not self._ready.is_set():
                self._ready.set()
        finally:
            self.session = None
