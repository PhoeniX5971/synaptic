"""Bounded all-or-nothing tool catalog discovery."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from .errors import MCPDiscoveryError
from .results import json_value


@dataclass(frozen=True)
class DiscoveryLimits:
    max_pages: int = 100
    max_tools: int = 10000
    max_bytes: int = 8 * 1024 * 1024
    timeout: float = 30.0

    def __post_init__(self):
        import math

        for value in (self.max_pages, self.max_tools, self.max_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Discovery counts must be positive integers")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Discovery timeout must be positive and finite")


async def discover(session, limits: DiscoveryLimits) -> list:
    from jsonschema.validators import validator_for  # type: ignore[import-untyped]

    tools: list[Any] = []
    cursors: set[str] = set()
    names: set[str] = set()
    cursor, size = None, 0
    async with asyncio.timeout(limits.timeout):
        for _ in range(limits.max_pages):
            page = await session.list_tools(cursor=cursor)
            data = json_value(page)
            size += len(json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            if size > limits.max_bytes:
                raise MCPDiscoveryError("Discovery byte limit exceeded")
            incoming = getattr(page, "tools", None)
            if not isinstance(incoming, list):
                raise MCPDiscoveryError("Discovery tools must be a list")
            if len(tools) + len(incoming) > limits.max_tools:
                raise MCPDiscoveryError("Discovery tool limit exceeded")
            for tool in incoming:
                name = getattr(tool, "name", None)
                if not isinstance(name, str) or not name or name in names:
                    raise MCPDiscoveryError("Invalid or duplicate discovery tool name")
                schema = getattr(tool, "inputSchema", getattr(tool, "input_schema", None))
                if not isinstance(schema, dict):
                    raise MCPDiscoveryError("Tool input schema must be an object")
                validator_for(schema).check_schema(schema)
                names.add(name)
            tools.extend(incoming)
            cursor = data.get("nextCursor", data.get("next_cursor"))
            if cursor is None:
                return tools
            if not isinstance(cursor, str) or cursor in cursors:
                raise MCPDiscoveryError("Invalid or repeated discovery cursor")
            cursors.add(cursor)
    raise MCPDiscoveryError("Discovery page limit exceeded")
