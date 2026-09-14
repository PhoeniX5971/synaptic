"""Public error categories at MCP protocol and transport boundaries."""

from typing import Any


class MCPTransportError(ConnectionError):
    """The connection failed; a dispatched tool may already have had effects."""


class MCPProtocolError(RuntimeError):
    """A JSON-RPC error, distinct from a tool's isError result."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class MCPDiscoveryError(ValueError):
    """A discovery refresh was rejected; no partial catalog was published."""


def classify_error(error: Exception) -> Exception:
    from mcp.shared.exceptions import McpError
    from mcp.types import CONNECTION_CLOSED
    import anyio
    import httpx
    from .results import json_value

    if isinstance(error, McpError):
        if error.error.code in (CONNECTION_CLOSED, 408):
            return MCPTransportError("MCP connection closed or timed out")
        return MCPProtocolError(error.error.code, error.error.message, json_value(error.error.data))
    if isinstance(
        error,
        (
            OSError,
            TimeoutError,
            httpx.TransportError,
            anyio.BrokenResourceError,
            anyio.ClosedResourceError,
        ),
    ):
        return MCPTransportError("MCP transport failed; tool completion may be unknown")
    return error
