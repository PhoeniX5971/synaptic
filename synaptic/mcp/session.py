"""MCP session API: explicit connection ownership and atomic catalog refresh."""

import asyncio
import copy
import time
from typing import Any

from ..logging import _bus
from ._connection import Connection, ConnectionState
from ._discovery import DiscoveryLimits, discover
from .errors import MCPTransportError, classify_error
from .servers import MCPServer


class MCPSession(Connection):
    def __init__(
        self,
        server: MCPServer,
        *,
        discovery_limits: DiscoveryLimits | None = None,
        **connection_options,
    ) -> None:
        super().__init__(server, **connection_options)
        self.discovery_limits = discovery_limits or DiscoveryLimits()
        self._tools: list[Any] | None = None
        self._catalog_lock = asyncio.Lock()

    async def list_tools(self, refresh: bool = False) -> list[Any]:
        """Discover every page; errors retain the prior catalog without partial publication.

        Limits bound decoded catalog bytes/counts and elapsed discovery time. SDK
        transport buffering is not a wire-level memory limit. Returned snapshots
        are deep copies so callers cannot mutate the cached schema definitions.
        """
        async with asyncio.timeout(self.discovery_limits.timeout):
            async with self._catalog_lock:
                await self.connect()
                if self._tools is None or refresh:
                    session = self.session
                    tools = await discover(session, self.discovery_limits)
                    if self.session is not session:
                        raise MCPTransportError("MCP connection changed during discovery")
                    self._tools = copy.deepcopy(tools)
                return copy.deepcopy(self._tools)

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Call once; protocol/transport exceptions stay distinct from tool failures."""
        await self.connect()
        dispatch = (self._attempt, self.session, self._task)
        t0 = time.monotonic()
        try:
            async with asyncio.timeout(self.request_timeout):
                result = await dispatch[1].call_tool(name, arguments=arguments)
        except Exception as exc:
            error = classify_error(exc)
            if isinstance(error, MCPTransportError) and all(
                old is current
                for old, current in zip(dispatch, (self._attempt, self.session, self._task))
            ):
                self.session = None
                self.state = ConnectionState.FAILED
                self._error = error
                if self._task is not None:
                    self._task.cancel()
            if error is exc:
                raise
            raise error from exc
        _bus.emit(
            {
                "type": "mcp.tool",
                "server": self.server.name,
                "attempt_id": id(dispatch[0]),
                "name": name,
                "args": arguments,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }
        )
        return result

    async def close(self) -> None:
        self._tools = None
        await super().close()
