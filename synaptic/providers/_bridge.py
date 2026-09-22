"""Backpressured bridge for SDKs exposing synchronous iterators only."""

import asyncio
import concurrent.futures
import threading


async def stream(factory, scheduler, capacity=16):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[object] = asyncio.Queue(maxsize=capacity)
    stop = threading.Event()
    sentinel = object()

    def put(value):
        future = asyncio.run_coroutine_threadsafe(queue.put(value), loop)
        while not stop.is_set():
            try:
                future.result(timeout=0.05)
                return
            except concurrent.futures.TimeoutError:
                continue
        future.cancel()

    def produce():
        iterator = None
        try:
            iterator = iter(factory())
            for item in iterator:
                if stop.is_set():
                    break
                put(item)
        except Exception as exc:
            if not stop.is_set():
                put(exc)
        finally:
            if iterator is not None and hasattr(iterator, "close"):
                iterator.close()
            if not stop.is_set():
                put(sentinel)

    task = asyncio.create_task(scheduler.run(produce))
    try:
        while True:
            receive = asyncio.create_task(queue.get())
            try:
                done, _ = await asyncio.wait(
                    (receive, task), return_when=asyncio.FIRST_COMPLETED
                )
                if task in done:
                    await task
                    if receive not in done and queue.empty():
                        return
                item = await receive
            finally:
                if not receive.done():
                    receive.cancel()
                    try:
                        await receive
                    except asyncio.CancelledError:
                        pass
            if item is sentinel:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
