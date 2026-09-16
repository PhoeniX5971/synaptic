"""Opt-in bounded thread execution; cancellation never asserts absence of effects."""

import asyncio
import contextvars
import inspect
from concurrent.futures import ThreadPoolExecutor


class ThreadScheduler:
    def __init__(self, workers=4):
        if not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be positive")
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="synaptic")
        self._slots = asyncio.Semaphore(workers)
        self._closed = False

    async def run(self, function, *args):
        if self._closed:
            raise RuntimeError("Scheduler is closed")
        await self._slots.acquire()
        if self._closed:
            self._slots.release()
            raise RuntimeError("Scheduler is closed")
        loop = asyncio.get_running_loop()
        try:
            future = self._pool.submit(contextvars.copy_context().run, function, *args)
        except BaseException:
            self._slots.release()
            raise

        def finished(_):
            if not loop.is_closed():
                loop.call_soon_threadsafe(self._slots.release)

        future.add_done_callback(finished)
        # Capacity stays occupied until actual completion, even if this caller cancels.
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(wrapped)

    def close(self):
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)


async def call_async(function, *args, scheduler=None):
    value = (
        await scheduler.run(function, *args)
        if scheduler and not inspect.iscoroutinefunction(function)
        else function(*args)
    )
    return await value if inspect.isawaitable(value) else value
