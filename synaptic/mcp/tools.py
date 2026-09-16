"""Canonical MCP identities and bounded, collision-checked provider aliases."""

import hashlib
import json
import re
from dataclasses import dataclass

from ..core.tool import Tool
from .results import json_value, normalize_result


@dataclass(frozen=True)
class ToolIdentity:
    server: str
    tool: str


def tool_name(server_name: str, mcp_name: str) -> str:
    simple = f"{server_name}__{mcp_name}"
    if (
        len(simple) <= 64
        and "__" not in server_name
        and "__" not in mcp_name
        and re.fullmatch(r"[a-zA-Z0-9_-]+", simple)
    ):
        return simple
    canonical = json.dumps([server_name, mcp_name], ensure_ascii=False).encode()
    return "mcp_" + hashlib.sha256(canonical).hexdigest()[:48]


def make_tool(server_name, mcp_tool, session, *, invoke=None):
    definition = json_value(mcp_tool)
    name = definition["name"]
    schema = definition.get("inputSchema", definition.get("input_schema", {}))
    from jsonschema.validators import validator_for

    validator_for(schema).check_schema(schema)

    output_schema = definition.get("outputSchema", definition.get("output_schema"))
    if output_schema is not None:
        validator_for(output_schema).check_schema(output_schema)

    async def call(**arguments):
        result = await (invoke(name, arguments) if invoke else session.call_tool(name, arguments))
        normalized = normalize_result(result, provenance={"server": server_name, "tool": name})
        if output_schema is not None and not normalized["is_error"]:
            if not normalized["has_structured_content"]:
                raise ValueError("MCP output schema requires structured content")
            validator_for(output_schema)(output_schema).validate(normalized["structured_content"])
        return normalized

    alias = tool_name(server_name, name)
    return Tool(
        alias,
        {"name": alias, "description": definition.get("description") or "", "parameters": schema},
        call,
        add_to_registry=False,
        identity=ToolIdentity(server_name, name),
        metadata=definition,
    )
