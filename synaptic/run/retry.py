"""Bounded retries for safe generation; visible output and tool effects are barriers."""

import asyncio
import time
from functools import wraps

from ._retry_policy import RetryPolicy


def retry(max_retries=3, delay=0.5, exceptions=(Exception,), **options):
    """Generic caller-responsible decorator. Repetition must be safe for this callable.

    Sync deadlines bound scheduling, not an already-running blocking function.
    Async attempts have a total deadline. Cancellation never proves no side effect.
    """
    policy = RetryPolicy(max_retries, delay, exceptions=exceptions, **options)

    def decorate(function):
        if asyncio.iscoroutinefunction(function):

            @wraps(function)
            async def async_wrapper(*args, **kwargs):
                async with asyncio.timeout(policy.deadline):
                    start = time.monotonic()
                    for attempt in range(policy.retries + 1):
                        try:
                            return await function(*args, **kwargs)
                        except Exception as error:
                            pause = policy.pause(
                                attempt, error, policy.deadline - (time.monotonic() - start)
                            )
                            if pause is None:
                                raise
                            await asyncio.sleep(pause)

            return async_wrapper

        @wraps(function)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            for attempt in range(policy.retries + 1):
                try:
                    return function(*args, **kwargs)
                except Exception as error:
                    pause = policy.pause(
                        attempt, error, policy.deadline - (time.monotonic() - start)
                    )
                    if pause is None:
                        raise
                    time.sleep(pause)

        return wrapper

    return decorate


class Retry:
    def __init__(
        self, model, max_retries=3, delay=0.5, exceptions=None, retry_safe=False, **options
    ):
        self._model = model
        self._retry_safe = retry_safe
        self._policy = RetryPolicy(max_retries, delay, exceptions=exceptions, **options)

    def _can_retry(self, kwargs):
        autorun = kwargs.get("autorun")
        if autorun is None:
            autorun = getattr(self._model, "autorun", None)
        return autorun is False or (autorun is not True and self._retry_safe)

    def invoke(self, *args, **kwargs):
        if not self._can_retry(kwargs):
            return self._model.invoke(*args, **kwargs)
        return self._wrap(self._model.invoke)(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        if not self._can_retry(kwargs):
            return await self._model.ainvoke(*args, **kwargs)
        return await self._wrap(self._model.ainvoke)(*args, **kwargs)

    def _wrap(self, function):
        p = self._policy
        return retry(
            p.retries,
            p.delay,
            exceptions=None,
            multiplier=p.multiplier,
            max_delay=p.max_delay,
            jitter=p.jitter,
            deadline=p.deadline,
            predicate=p.predicate,
        )(function)

    async def astream(self, *args, **kwargs):
        p = self._policy
        start = time.monotonic()
        emitted = False
        async with asyncio.timeout(p.deadline):
            for attempt in range(p.retries + 1):
                source = self._model.astream(*args, **kwargs)
                try:
                    async for chunk in source:
                        emitted = True
                        yield chunk
                    return
                except Exception as error:
                    if emitted or not self._can_retry(kwargs):
                        raise
                    pause = p.pause(attempt, error, p.deadline - (time.monotonic() - start))
                    if pause is None:
                        raise
                    await asyncio.sleep(pause)
                finally:
                    close = getattr(source, "aclose", None)
                    if close:
                        await close()

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._model, name, value)
