"""Coalesced, bounded connection ownership for SDK/AnyIO transports."""

import asyncio
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..logging import _bus
from ._transport import open_transport
from .servers import MCPServer, StdioMCPServer


class ConnectionState(StrEnum):
    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass
class _Attempt:
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    error: BaseException | None = None
    waiters: int = 0


class Connection:
    def __init__(
        self,
        server: MCPServer,
        *,
        connect_timeout: float = 10.0,
        shutdown_timeout: float = 5.0,
        health_interval: float = 15.0,
        request_timeout: float = 30.0,
    ) -> None:
        for value in (connect_timeout, shutdown_timeout, health_interval, request_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("MCP timeouts must be positive and finite")
        self.server = server
        self.connect_timeout, self.shutdown_timeout = connect_timeout, shutdown_timeout
        self.health_interval, self.request_timeout = health_interval, request_timeout
        self.state = ConnectionState.NEW
        self.session: Any = None
        self._task: asyncio.Task | None = None
        self._attempt: _Attempt | None = None
        self._ready: asyncio.Event | None = None
        self._error: BaseException | None = None
        self.protocol_version: str | None = None

    async def connect(self) -> None:
        """Coalesce startup; failed attempts require explicit reconnect or close."""
        if self.state == ConnectionState.CLOSING:
            raise RuntimeError("MCP session is closing")
        if self.state == ConnectionState.CONNECTED:
            if self._task is not None and not self._task.done() and self.session is not None:
                return
            self.state = ConnectionState.FAILED
            self.session = None
        if self.state == ConnectionState.FAILED:
            raise self._error or RuntimeError("MCP owner terminated; reconnect required")
        if self.state in (ConnectionState.NEW, ConnectionState.CLOSED):
            attempt = self._attempt = _Attempt()
            self._ready, self._error = attempt.ready, None
            self.state = ConnectionState.CONNECTING
            self._task = asyncio.create_task(self._owner(attempt))
            self._task.add_done_callback(lambda task: self._owner_done(attempt, task))
        assert self._attempt is not None
        attempt = self._attempt
        attempt.waiters += 1
        try:
            await asyncio.wait_for(attempt.ready.wait(), self.connect_timeout)
            if attempt.error is not None:
                raise attempt.error
            if self.state != ConnectionState.CONNECTED or self.session is None:
                raise self._error or RuntimeError("MCP connection did not become live")
        finally:
            attempt.waiters -= 1
            if attempt.waiters == 0 and self.state == ConnectionState.CONNECTING:
                # No remaining caller owns this startup. Do not leak its transport.
                assert self._task is not None
                self._task.cancel()

    async def reconnect(self) -> None:
        """Close the previous owner before starting one bounded fresh attempt."""
        await self.close()
        await self.connect()

    async def close(self) -> None:
        """Idempotent teardown; quarantine an owner that exceeds shutdown timeout."""
        task = self._task
        self.session = None
        if task is None or task.done():
            self.state = ConnectionState.CLOSED
            return
        if self.state != ConnectionState.CLOSING:
            self.state = ConnectionState.CLOSING
            assert self._attempt is not None
            self._attempt.error = self._attempt.error or RuntimeError(
                "MCP session closed during startup"
            )
            self._attempt.ready.set()
            if not task.cancelling():
                task.cancel()
        done, _ = await asyncio.wait({task}, timeout=self.shutdown_timeout)
        if not done:
            raise TimeoutError("MCP shutdown timed out; owner remains quarantined")
        if self._task is task:
            self.state = ConnectionState.CLOSED
        _bus.emit({"type": "mcp.disconnect", "server": self.server.name})

    async def _owner(self, attempt: _Attempt) -> None:
        try:
            if isinstance(self.server, StdioMCPServer):
                await self._run_stdio()
            else:
                await self._run_http()
            attempt.error = self._error or RuntimeError("MCP transport owner terminated")
        except asyncio.CancelledError:
            attempt.error = attempt.error or self._error or RuntimeError("MCP connection cancelled")
        except Exception as exc:
            attempt.error = exc
        finally:
            self._error = attempt.error
            self.session = None
            self.state = (
                ConnectionState.CLOSED
                if self.state == ConnectionState.CLOSING
                else ConnectionState.FAILED
            )
            attempt.ready.set()

    def _owner_done(self, attempt: _Attempt, task: asyncio.Task) -> None:
        # A task cancelled before its coroutine starts never executes its finally.
        if task.cancelled() and not attempt.ready.is_set():
            attempt.error = RuntimeError("MCP startup cancelled before transport setup")
            if self._attempt is attempt:
                self._error = attempt.error
                self.state = ConnectionState.FAILED
                self.session = None
            attempt.ready.set()

    async def _run_stdio(self) -> None:
        await self._serve()

    async def _run_http(self) -> None:
        await self._serve()

    async def _serve(self) -> None:
        async with open_transport(self.server) as (read, write):
            from mcp import ClientSession

            async with ClientSession(read, write) as session:
                async with asyncio.timeout(self.connect_timeout):
                    initialized = await session.initialize()
                self.protocol_version = initialized.protocolVersion
                self.session = session
                self.state = ConnectionState.CONNECTED
                assert self._ready is not None
                self._ready.set()
                _bus.emit({"type": "mcp.connect", "server": self.server.name})
                while True:
                    await asyncio.sleep(self.health_interval)
                    async with asyncio.timeout(self.request_timeout):
                        await session.send_ping()
