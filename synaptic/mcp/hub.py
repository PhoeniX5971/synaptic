import asyncio
from uuid import uuid4
from typing import Dict, Iterable, List, Optional

from ..core.tool import Tool, ToolRegistry
from .servers import MCPServer
from .session import MCPSession
from .tools import make_tool
from ._hub_runtime import HubRuntime


class MCPHub:
    def __init__(
        self,
        servers: Iterable[MCPServer],
        registry: Optional[ToolRegistry] = None,
        *,
        concurrency=4,
        max_connections=16,
        max_servers=128,
        timeout=30.0,
        idle_timeout=300.0,
        reconnect_backoff=1.0,
    ):
        if isinstance(max_servers, bool) or not isinstance(max_servers, int) or max_servers < 1:
            raise ValueError("max_servers must be a positive integer")
        self.registry = registry or ToolRegistry()
        self.sessions: dict[str, MCPSession] = {}
        self.loaded: dict[str, list[str]] = {}
        self._id = uuid4().hex
        self._locks = {}
        self._versions: dict[str, int] = {}
        for server in servers:
            if server.name in self.sessions or len(self.sessions) >= max_servers:
                raise ValueError("Duplicate server or maximum server count exceeded")
            self.sessions[server.name] = MCPSession(server)
            self._locks[server.name] = asyncio.Lock()
        self.runtime = HubRuntime(
            self.sessions,
            concurrency=concurrency,
            max_connections=max_connections,
            timeout=timeout,
            idle_timeout=idle_timeout,
            reconnect_backoff=reconnect_backoff,
        )

    async def load_server(self, name: str, refresh: bool = False) -> List[str]:
        session = self._session(name)
        # Tests/integrations may replace session implementations before first load.
        self.runtime.sessions = self.sessions
        async with self._locks[name]:
            version = self._versions.get(name, 0)
            async with self.runtime.operation(name):
                tools = await session.list_tools(refresh=refresh)

                async def invoke(tool, arguments):
                    return await self.runtime.call(name, tool, arguments)

                replacements = [make_tool(name, tool, session, invoke=invoke) for tool in tools]
                names = [tool.name for tool in replacements]
                if (
                    self.runtime.closed
                    or name in self.runtime.draining
                    or version != self._versions.get(name, 0)
                ):
                    raise RuntimeError("Catalog changed or closed during refresh")

                def publish():
                    self.loaded = {**self.loaded, name: names}

                self.registry.replace_owned((self._id, name), replacements, publish=publish)
                return list(names)

    async def load_all(self) -> Dict[str, List[str] | Exception]:
        """Bounded workers; per-server exceptions are returned alongside successful catalogs."""
        return await self.runtime.map_load(list(self.sessions), self.load_server)

    def unload_server(self, name: str) -> None:
        """Remove discovery visibility only; existing calls and connection remain owned."""
        self._versions[name] = self._versions.get(name, 0) + 1
        if name in self.loaded:

            def publish():
                self.loaded = {key: value for key, value in self.loaded.items() if key != name}

            self.registry.replace_owned((self._id, name), [], publish=publish)

    def unload_all(self) -> None:
        for name in list(self.loaded):
            self.unload_server(name)

    async def disconnect_server(self, name):
        """Drain active calls and close transport, retaining catalog visibility."""
        self.runtime.sessions = self.sessions
        await self.runtime.disconnect(name)

    async def remove_server(self, name):
        """Drain, disconnect, remove registrations/config and invalidate stale wrappers."""
        async with self._locks[name]:
            await self.disconnect_server(name)
            self.unload_server(name)
            self.runtime.draining.add(name)
            self.sessions.pop(name)

    async def close(self):
        self.runtime.closed = True
        self.unload_all()
        self.registry.replace_owned((self._id, None), [])
        results = await self.runtime.map_load(list(self.sessions), self.disconnect_server)
        errors = [result for result in results.values() if isinstance(result, Exception)]
        if errors:
            raise ExceptionGroup("MCP close errors", errors)

    def resolve_alias(self, alias):
        """Reverse a provider alias through the current immutable definition snapshot."""
        return self.registry.all()[alias].identity

    def enable_routing(self) -> List[Tool]:
        tools = [self._list_servers_tool(), self._load_tool(), self._unload_tool()]
        self.registry.replace_owned((self._id, None), tools)
        return tools

    def _session(self, name: str) -> MCPSession:
        if name not in self.sessions:
            raise ValueError(f"Unknown MCP server: {name}")
        return self.sessions[name]

    def _list_servers_tool(self) -> Tool:
        async def mcp_list_servers() -> dict:
            return {"servers": list(self.sessions), "loaded": list(self.loaded)}

        return Tool(
            name="mcp_list_servers",
            declaration=_decl("mcp_list_servers", "List MCP servers."),
            function=mcp_list_servers,
            add_to_registry=False,
        )

    def _load_tool(self) -> Tool:
        async def mcp_load_server(name: str) -> dict:
            return {"server": name, "tools": await self.load_server(name)}

        return Tool(
            name="mcp_load_server",
            declaration=_decl("mcp_load_server", "Load one MCP server.", True),
            function=mcp_load_server,
            add_to_registry=False,
        )

    def _unload_tool(self) -> Tool:
        async def mcp_unload_server(name: str) -> dict:
            self.unload_server(name)
            return {"server": name, "unloaded": True}

        return Tool(
            "mcp_unload_server",
            declaration=_decl("mcp_unload_server", "Unload one MCP server.", True),
            function=mcp_unload_server,
            add_to_registry=False,
        )


def _decl(name: str, description: str, has_server: bool = False) -> dict:
    properties = {}
    if has_server:
        properties["name"] = {"type": "string", "description": "MCP server name"}
    params = {"type": "object", "properties": properties}
    if has_server:
        params["required"] = ["name"]
    return {"name": name, "description": description, "parameters": params}
