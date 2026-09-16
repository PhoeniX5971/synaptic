"""Bounded hub resources and call draining; catalogs do not own connections."""

import asyncio
import math
import time
from contextlib import asynccontextmanager


class HubRuntime:
    def __init__(
        self,
        sessions,
        *,
        concurrency=4,
        max_connections=16,
        timeout=30.0,
        idle_timeout=300.0,
        reconnect_backoff=1.0,
    ):
        if (
            any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in (concurrency, max_connections)
            )
            or not all(math.isfinite(v) for v in (timeout, idle_timeout, reconnect_backoff))
            or concurrency < 1
            or max_connections < 1
            or min(timeout, idle_timeout, reconnect_backoff) <= 0
        ):
            raise ValueError("Hub resource limits must be positive")
        self.sessions = sessions
        self.concurrency, self.max_connections = concurrency, max_connections
        self.timeout, self.idle_timeout = timeout, idle_timeout
        self.reconnect_backoff = reconnect_backoff
        self.slots = asyncio.Semaphore(concurrency)
        self.lock = asyncio.Lock()
        self.connected, self.draining = set(), set()
        self.active, self.used, self.failures = {}, {}, {}
        self.closed = False

    async def reserve(self, name):
        async with self.lock:
            if self.closed or name in self.draining:
                raise RuntimeError("MCP server is draining or closed")
            now = time.monotonic()
            if now < self.failures.get(name, 0):
                raise ConnectionError("MCP server reconnect backoff is active")
            for old in list(self.connected):
                if (
                    old != name
                    and not self.active.get(old)
                    and now - self.used.get(old, now) >= self.idle_timeout
                ):
                    await self.sessions[old].close()
                    self.connected.discard(old)
            if name not in self.connected and len(self.connected) >= self.max_connections:
                idle = [old for old in self.connected if not self.active.get(old)]
                if not idle:
                    raise RuntimeError("MCP connection capacity reached")
                old = min(idle, key=lambda item: self.used.get(item, 0))
                await self.sessions[old].close()
                self.connected.discard(old)
            self.connected.add(name)
            self.used[name] = now

    @asynccontextmanager
    async def operation(self, name):
        async with asyncio.timeout(self.timeout):
            async with self.slots:
                await self.reserve(name)
                self.active[name] = self.active.get(name, 0) + 1
                try:
                    yield
                except (ConnectionError, TimeoutError):
                    self.failures[name] = time.monotonic() + self.reconnect_backoff
                    raise
                finally:
                    self.active[name] -= 1
                    self.used[name] = time.monotonic()

    async def call(self, name, tool, arguments):
        async with self.operation(name):
            session = self.sessions[name]
            if getattr(session, "state", None) == "failed":
                await session.reconnect()
            return await session.call_tool(tool, arguments)

    async def disconnect(self, name):
        self.draining.add(name)
        errors: list[Exception] = []
        try:
            try:
                async with asyncio.timeout(self.timeout):
                    while self.active.get(name):
                        await asyncio.sleep(0.01)
            except TimeoutError as error:
                errors.append(error)
            try:
                async with asyncio.timeout(self.timeout):
                    await self.sessions[name].close()
                self.connected.discard(name)
            except Exception as error:
                errors.append(error)
            if errors:
                raise ExceptionGroup("MCP drain/close errors", errors)
        finally:
            self.draining.discard(name)

    async def map_load(self, names, load):
        queue: asyncio.Queue[str] = asyncio.Queue()
        for name in names:
            queue.put_nowait(name)
        results = {}

        async def worker():
            while not queue.empty():
                name = queue.get_nowait()
                try:
                    results[name] = await load(name)
                except Exception as exc:
                    results[name] = exc

        async with asyncio.TaskGroup() as group:
            for _ in range(min(self.concurrency, len(names))):
                group.create_task(worker())
        return results
