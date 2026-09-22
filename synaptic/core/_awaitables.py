"""Shared boundary for synchronous APIs that must not return async work."""

import inspect
from typing import Any


def require_sync(value: Any, message: str) -> Any:
    """Reject awaitables and close newly returned coroutines without running them.

    Futures and custom awaitables remain owned by the caller; rejecting one does
    not cancel it or prove its underlying work has stopped.
    """
    if inspect.isawaitable(value):
        if inspect.iscoroutine(value):
            value.close()
        raise RuntimeError(message)
    return value
