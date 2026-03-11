import asyncio
import time
from functools import wraps
from typing import Tuple, Type


def retry(
    max_retries: int = 3,
    delay: float = 0.5,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator that retries a sync or async function on failure.

    Usage:
        @retry(max_retries=3, delay=1.0)
        def call():
            return model.invoke("...")

        @retry(max_retries=3)
        async def acall():
            return await model.ainvoke("...")
    """

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
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        self._model = model
        self._max_retries = max_retries
        self._delay = delay
        self._exceptions = exceptions

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------
    def invoke(self, *args, **kwargs):
        last_exc: Exception
        for attempt in range(self._max_retries + 1):
            try:
                return self._model.invoke(*args, **kwargs)
            except self._exceptions as e:
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
                last_exc = e
                if attempt < self._max_retries:
                    await asyncio.sleep(self._delay)
        raise last_exc  # type: ignore[possibly-undefined]

    async def astream(self, *args, **kwargs):
        """
        Retries the entire stream from scratch on failure.
        If chunks were already yielded before the error, they won't be re-sent.
        Ideal for catching connection/auth failures before streaming begins.
        """
        last_exc: Exception
        for attempt in range(self._max_retries + 1):
            try:
                async for chunk in self._model.astream(*args, **kwargs):
                    yield chunk
                return
            except self._exceptions as e:
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
