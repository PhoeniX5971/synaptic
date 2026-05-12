from typing import Any, Callable, Dict

from ..core.tool import Tool
from .results import normalize_result
from .session import MCPSession


def tool_name(server_name: str, mcp_name: str) -> str:
    return f"{server_name}__{mcp_name}"


def make_tool(server_name: str, mcp_tool: Any, session: MCPSession) -> Tool:
    name = getattr(mcp_tool, "name")
    description = getattr(mcp_tool, "description", "") or ""
    schema = getattr(mcp_tool, "inputSchema", None) or getattr(mcp_tool, "input_schema", None)
    schema = schema or {"type": "object", "properties": {}}
    call = _make_call(name, session)
    return Tool(
        name=tool_name(server_name, name),
        declaration={
            "name": tool_name(server_name, name),
            "description": description,
            "parameters": schema,
        },
        function=call,
        add_to_registry=False,
    )


def _make_call(name: str, session: MCPSession) -> Callable[..., Any]:
    async def call(**kwargs: Any) -> Any:
        result = await session.call_tool(name, kwargs)
        return normalize_result(result)

    return call
