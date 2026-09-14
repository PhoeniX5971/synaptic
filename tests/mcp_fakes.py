"""Short-deadline session factory for in-process lifecycle tests."""

from synaptic.mcp import StdioMCPServer
from synaptic.mcp.session import MCPSession


def make_session():
    return MCPSession(
        StdioMCPServer("test", "unused"),
        connect_timeout=0.1,
        shutdown_timeout=0.1,
        health_interval=0.05,
        request_timeout=0.1,
    )
