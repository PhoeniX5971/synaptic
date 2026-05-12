from typing import Dict, Iterable, List, Optional

from ..core.tool import Tool, ToolRegistry
from .servers import MCPServer
from .session import MCPSession
from .tools import make_tool


class MCPHub:
    def __init__(
        self,
        servers: Iterable[MCPServer],
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.sessions: Dict[str, MCPSession] = {}
        self.loaded: Dict[str, List[str]] = {}
        for server in servers:
            if server.name in self.sessions:
                raise ValueError(f"Duplicate MCP server name: {server.name}")
            self.sessions[server.name] = MCPSession(server)

    async def load_server(self, name: str, refresh: bool = False) -> List[str]:
        session = self._session(name)
        tools = await session.list_tools(refresh=refresh)
        self.unload_server(name)
        synaptic_tools = [make_tool(name, mcp_tool, session) for mcp_tool in tools]
        self.registry.register_many(synaptic_tools)
        names = [tool.name for tool in synaptic_tools]
        self.loaded[name] = names
        return names

    async def load_all(self) -> Dict[str, List[str]]:
        loaded = {}
        for name in self.sessions:
            loaded[name] = await self.load_server(name)
        return loaded

    def unload_server(self, name: str) -> None:
        if name in self.loaded:
            self.loaded.pop(name)
            self.registry.unregister_prefix(f"{name}__")

    def unload_all(self) -> None:
        for name in list(self.loaded):
            self.unload_server(name)

    async def close(self) -> None:
        self.unload_all()
        for session in self.sessions.values():
            await session.close()

    def enable_routing(self) -> List[Tool]:
        tools = [self._list_servers_tool(), self._load_tool(), self._unload_tool()]
        for tool in tools:
            self.registry.register(tool)
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
