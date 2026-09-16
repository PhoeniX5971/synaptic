"""Observational events with explicit failure policy and disposable subscriptions."""

import copy
import warnings

from ..core._awaitables import require_sync
from ..core._registry import Subscription
from ..run.scheduling import call_async


class EventBus:
    def __init__(self, *, errors="warn", on_error=None, scheduler=None):
        if errors not in ("warn", "raise"):
            raise ValueError("errors must be warn or raise")
        self._handlers = {}
        self.errors, self.on_error, self.scheduler = errors, on_error, scheduler

    def on(self, event, fn):
        handlers = self._handlers.setdefault(event, [])
        handlers.append(fn)
        return Subscription(lambda: handlers.remove(fn) if fn in handlers else None)

    def _failed(self, exc):
        if self.errors == "raise":
            raise exc
        try:
            if self.on_error:
                require_sync(self.on_error(exc), "Error handlers must be synchronous")
            else:
                warnings.warn("Event observer failed: " + type(exc).__name__, RuntimeWarning)
        except Exception:
            pass

    def emit(self, event, *args):
        for fn in tuple(self._handlers.get(event, [])):
            try:
                result = fn(*copy.deepcopy(args))
                require_sync(result, "Async event handlers require aemit()")
            except RuntimeError as exc:
                if str(exc) == "Async event handlers require aemit()":
                    raise
                self._failed(exc)
            except Exception as exc:
                self._failed(exc)

    async def aemit(self, event, *args):
        for fn in tuple(self._handlers.get(event, [])):
            try:
                await call_async(fn, *copy.deepcopy(args), scheduler=self.scheduler)
            except Exception as exc:
                self._failed(exc)

    def close(self):
        self._handlers.clear()
