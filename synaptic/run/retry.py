import asyncio
import time
import math
from functools import wraps
from typing import Tuple, Type


def _validate(max_retries: int, delay: float) -> None:
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("delay must be finite and nonnegative")


def retry(
    max_retries: int = 3,
    delay: float = 0.5,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator that retries a sync or async function on failure.

    The caller must ensure repeated execution is safe; effects are not rolled back.

    Usage:
        @retry(max_retries=3, delay=1.0)
        def call():
            return model.invoke("...")

        @retry(max_retries=3)
        async def acall():
            return await model.ainvoke("...")
    """

    _validate(max_retries, delay)

    def decorator(fn):
        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                last_exc: Exception
                for attempt in range(max_retries + 1):
                    try:
                        return await fn(*args, **kwargs)
                    except exceptions as e:
                        last_exc = e
                        if attempt < max_retries:
                            await asyncio.sleep(delay)
                raise last_exc  # type: ignore[possibly-undefined]

            return async_wrapper

        @wraps(fn)
        def sync_wrapper(*args, **kwargs):
            last_exc: Exception
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        time.sleep(delay)
            raise last_exc  # type: ignore[possibly-undefined]

        return sync_wrapper

    return decorator


class Retry:
    """
    Wraps a Model instance to add retry logic to invoke / ainvoke / astream.

    Usage:
        model = Model(provider=Provider.OPENAI, ...)
        robust = Retry(model, max_retries=3, delay=1.0)

        response = robust.invoke("Hello")
        response = await robust.ainvoke("Hello")
        async for chunk in robust.astream("Hello"):
            print(chunk.text, end="")

    Any attribute not found on Retry is forwarded to the wrapped model,
    so Retry is a transparent proxy (e.g. robust.history, robust.instructions).
    """

    def __init__(
        self,
        model,
        max_retries: int = 3,
        delay: float = 0.5,
        exceptions: Tuple[Type[Exception], ...] = (ConnectionError, TimeoutError),
        retry_safe: bool = False,
    ):
        _validate(max_retries, delay)
        self._retry_safe = retry_safe
        self._model = model
        self._max_retries = max_retries
        self._delay = delay
        self._exceptions = exceptions

    def _can_retry(self, kwargs) -> bool:
        autorun = kwargs.get("autorun")
        if autorun is None:
            autorun = getattr(self._model, "autorun", None)
        return autorun is False or (autorun is not True and self._retry_safe)

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------
    def invoke(self, *args, **kwargs):
        last_exc: Exception
        for attempt in range(self._max_retries + 1):
            try:
                return self._model.invoke(*args, **kwargs)
            except self._exceptions as e:
                if not self._can_retry(kwargs):
                    raise
                last_exc = e
                if attempt < self._max_retries:
                    time.sleep(self._delay)
        raise last_exc  # type: ignore[possibly-undefined]

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------
    async def ainvoke(self, *args, **kwargs):
        last_exc: Exception
        for attempt in range(self._max_retries + 1):
            try:
                return await self._model.ainvoke(*args, **kwargs)
            except self._exceptions as e:
                if not self._can_retry(kwargs):
                    raise
                last_exc = e
                if attempt < self._max_retries:
                    await asyncio.sleep(self._delay)
        raise last_exc  # type: ignore[possibly-undefined]

    async def astream(self, *args, **kwargs):
        """
        Retry transient failures only before the first yielded chunk.

        Tool-executing models are never transparently retried. Unknown producers
        require retry_safe=True, asserting no effects occur before a yield.
        After any visible chunk, the original error propagates without restart.
        """
        emitted = False
        last_exc: Exception
        for attempt in range(self._max_retries + 1):
            try:
                async for chunk in self._model.astream(*args, **kwargs):
                    emitted = True
                    yield chunk
                return
            except self._exceptions as e:
                if emitted or not self._can_retry(kwargs):
                    raise
                last_exc = e
                if attempt < self._max_retries:
                    await asyncio.sleep(self._delay)
        raise last_exc  # type: ignore[possibly-undefined]

    # Transparent proxy — forward everything else to the underlying model
    def __getattr__(self, name: str):
        return getattr(self._model, name)

    def __setattr__(self, name: str, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._model, name, value)
