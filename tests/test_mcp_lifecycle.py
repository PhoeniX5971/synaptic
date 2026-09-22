import asyncio
from types import SimpleNamespace

import pytest

from synaptic.mcp import ConnectionState
from tests.mcp_fakes import make_session


def test_exception_before_transport_releases_every_waiter():
    async def run():
        session = make_session()

        async def fail():
            raise ImportError("missing SDK")

        session._run_stdio = fail
        results = await asyncio.wait_for(
            asyncio.gather(*(session.connect() for _ in range(5)), return_exceptions=True), 1
        )
        assert all(isinstance(result, ImportError) for result in results)
        assert session.state == ConnectionState.FAILED
        await session.close()

    asyncio.run(run())


def test_one_cancelled_waiter_does_not_cancel_other_startup_callers():
    async def run():
        session = make_session()
        gate = asyncio.Event()

        async def start():
            await gate.wait()
            session.session = SimpleNamespace()
            session.state = ConnectionState.CONNECTED
            session._ready.set()
            await asyncio.Future()

        session._run_stdio = start
        first = asyncio.create_task(session.connect())
        second = asyncio.create_task(session.connect())
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        gate.set()
        await second
        assert session.state == ConnectionState.CONNECTED
        await session.close()
        await session.close()
        assert session.session is None

    asyncio.run(run())


def test_close_during_initialization_releases_waiters_and_allows_reconnect():
    async def run():
        session = make_session()
        started = asyncio.Event()

        async def hang():
            started.set()
            await asyncio.Future()

        session._run_stdio = hang
        connecting = asyncio.create_task(session.connect())
        await started.wait()
        await session.close()
        with pytest.raises(RuntimeError):
            await connecting

        async def success():
            session.session = SimpleNamespace()
            session.state = ConnectionState.CONNECTED
            session._ready.set()
            await asyncio.Future()

        session._run_stdio = success
        await session.reconnect()
        assert session.session is not None
        await session.close()

    asyncio.run(run())


def test_startup_timeout_and_owner_death_do_not_leave_live_state():
    async def run():
        session = make_session()

        async def hang():
            await asyncio.Future()

        session._run_stdio = hang
        with pytest.raises(TimeoutError):
            await session.connect()
        await session.close()
        assert session.session is None

        async def die():
            session.session = SimpleNamespace()
            session.state = ConnectionState.CONNECTED
            session._ready.set()
            await asyncio.sleep(0.01)
            raise ConnectionError("dead")

        session._run_stdio = die
        await session.connect()
        await asyncio.sleep(0.02)
        assert session.state == ConnectionState.FAILED
        with pytest.raises(ConnectionError):
            await session.connect()
        await session.close()

    asyncio.run(run())


def test_slow_shutdown_remains_quarantined_until_owner_exits():
    async def run():
        session = make_session()
        release = asyncio.Event()

        async def stubborn():
            session.session = SimpleNamespace()
            session.state = ConnectionState.CONNECTED
            session._ready.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                await release.wait()

        session._run_stdio = stubborn
        await session.connect()
        try:
            with pytest.raises(TimeoutError, match="quarantined"):
                await session.close()
            assert session.state == ConnectionState.CLOSING
            with pytest.raises(RuntimeError, match="closing"):
                await session.connect()
            assert session.session is None
        finally:
            release.set()
            await session.close()
        assert session.state == ConnectionState.CLOSED

    asyncio.run(run())


def test_cancelling_only_startup_waiter_releases_transport_owner():
    async def run():
        session = make_session()
        started = asyncio.Event()

        async def hang():
            started.set()
            await asyncio.Future()

        session._run_stdio = hang
        connecting = asyncio.create_task(session.connect())
        await started.wait()
        connecting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await connecting
        await session.close()
        assert session._task.done()
        assert session.session is None

    asyncio.run(run())


def test_concurrent_reconnect_does_not_replace_a_new_owner():
    async def run():
        session = make_session()
        starts = 0

        async def success():
            nonlocal starts
            starts += 1
            session.session = SimpleNamespace()
            session.state = ConnectionState.CONNECTED
            session._ready.set()
            await asyncio.Future()

        session._run_stdio = success
        await session.connect()
        await asyncio.gather(session.reconnect(), session.reconnect())
        assert session.state == ConnectionState.CONNECTED
        assert starts == 2
        await session.close()

    asyncio.run(run())
