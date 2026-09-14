"""Tests for MCP hub tool loading without a real MCP server."""
import asyncio
from types import SimpleNamespace

from synaptic.core.tool import ToolRegistry
from synaptic.mcp import MCPHub, StdioMCPServer


class FakeSession:
    def __init__(self):
        self.calls = []
        self.closed = False

    async def list_tools(self, refresh=False):
        return [
            SimpleNamespace(
                name="echo",
                description="Echo text",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        content = [SimpleNamespace(text=arguments["text"])]
        return SimpleNamespace(content=content, structuredContent=None, isError=False)

    async def close(self):
        self.closed = True


def make_hub():
    registry = ToolRegistry()
    hub = MCPHub([StdioMCPServer("small", "demo")], registry=registry)
    fake = FakeSession()
    hub.sessions["small"] = fake
    return hub, registry, fake


def test_load_server_registers_prefixed_tools():
    async def run():
        hub, registry, fake = make_hub()
        names = await hub.load_server("small")
        assert names == ["small__echo"]

        tool = registry.all()["small__echo"]
        assert tool.declaration["parameters"]["required"] == ["text"]
        assert await tool.run(text="hi") == "hi"
        assert fake.calls == [("echo", {"text": "hi"})]

    asyncio.run(run())


def test_unload_server_removes_loaded_tools():
    async def run():
        hub, registry, _ = make_hub()
        await hub.load_server("small")
        hub.unload_server("small")
        assert "small__echo" not in registry.all()

    asyncio.run(run())


def test_routing_is_opt_in():
    hub, registry, _ = make_hub()
    assert registry.all() == {}

    tools = hub.enable_routing()
    assert [tool.name for tool in tools] == [
        "mcp_list_servers",
        "mcp_load_server",
        "mcp_unload_server",
    ]
    assert "small__echo" not in registry.all()


def test_router_load_tool_loads_server_tools():
    async def run():
        hub, registry, _ = make_hub()
        hub.enable_routing()
        result = await registry.all()["mcp_load_server"].run(name="small")
        assert result == {"server": "small", "tools": ["small__echo"]}
        assert "small__echo" in registry.all()

    asyncio.run(run())
