"""Reject concurrent mutable model/history use while allowing an agent's nested calls."""

import asyncio
import threading
import inspect
from contextlib import contextmanager
from functools import wraps
from uuid import uuid4
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])
from ..logging import _bus


_lock = threading.RLock()


def current_owner():
    try:
        return asyncio.current_task() or threading.get_ident()
    except RuntimeError:
        return threading.get_ident()


@contextmanager
def operation(model, history=None):
    owner = current_owner()
    objects = [model, history if history is not None else getattr(model, "history", None)]
    acquired = []
    try:
        with _lock:
            for obj in objects:
                if obj is None:
                    continue
                active = getattr(obj, "_operation_owner", None)
                if active is not None and active != owner:
                    raise RuntimeError(
                        "Concurrent model/history use requires separate instances"
                    )
                if active is None:
                    obj._operation_owner = owner
                    acquired.append(obj)
        correlation = {} if _bus._correlation.get().get("run_id") else {"run_id": uuid4().hex}
        with _bus.current().scope(**correlation):
            yield
    finally:
        with _lock:
            for obj in acquired:
                obj._operation_owner = None


def guarded(function: F) -> F:
    @contextmanager
    def context(self):
        correlation = {"session_id": self.session.id} if hasattr(self, "session") else {}
        with _bus.current().scope(**correlation):
            with operation(
                self.model if hasattr(self, "session") else self,
                self.session.history if hasattr(self, "session") else None,
            ):
                yield

    if inspect.isasyncgenfunction(function):

        @wraps(function)
        async def stream_wrapper(self, *args, **kwargs):
            with context(self):
                source = function(self, *args, **kwargs)
                try:
                    async for item in source:
                        yield item
                finally:
                    await source.aclose()

        return cast(F, stream_wrapper)
    if asyncio.iscoroutinefunction(function):

        @wraps(function)
        async def async_wrapper(self, *args, **kwargs):
            with context(self):
                return await function(self, *args, **kwargs)

        return cast(F, async_wrapper)

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        with context(self):
            return function(self, *args, **kwargs)

    return cast(F, wrapper)
