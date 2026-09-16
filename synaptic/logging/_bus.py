"""Scoped best-effort diagnostics; application journals remain application-owned."""

import copy
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

from ._privacy import prepare

stream_chunks = False
_current = ContextVar("synaptic_diagnostics", default=None)
_correlation = ContextVar("synaptic_correlation", default={})


class DiagnosticBus:
    def __init__(self, *, payloads=False, secrets=(), redact=None, max_bytes=16384, on_error=None):
        if max_bytes < 256:
            raise ValueError("max_bytes must be at least 256")
        self.payloads, self.secrets, self.redact = payloads, tuple(secrets), redact
        self.max_bytes, self.on_error = max_bytes, on_error
        self._subscribers = []
        self.failures = 0

    def subscribe(self, fn):
        from ..core._registry import Subscription

        if fn not in self._subscribers:
            self._subscribers.append(fn)
        return Subscription(lambda: self.unsubscribe(fn))

    def unsubscribe(self, fn):
        if fn in self._subscribers:
            self._subscribers.remove(fn)

    def emit(self, event):
        if not self._subscribers:
            return
        candidate = {**_correlation.get(), **event, "ts": datetime.now(timezone.utc).isoformat()}
        try:
            candidate = prepare(candidate, self.payloads, self.secrets, self.redact, self.max_bytes)
        except Exception as exc:
            self._failure(exc)
            return
        for sink in tuple(self._subscribers):
            try:
                from ..core._awaitables import require_sync

                require_sync(
                    sink(copy.deepcopy(candidate)),
                    "Async sinks require an explicit delivery adapter",
                )
            except Exception as exc:
                self._failure(exc)

    def _failure(self, exc):
        from ..core._awaitables import require_sync

        self.failures += 1
        try:
            if self.on_error:
                require_sync(
                    self.on_error(type(exc).__name__), "Error handlers must be synchronous"
                )
            else:
                warnings.warn("Diagnostic sink failed: " + type(exc).__name__, RuntimeWarning)
        except Exception:
            pass

    @contextmanager
    def scope(self, **correlation):
        bus_token = _current.set(self)
        token = _correlation.set({**_correlation.get(), **correlation})
        try:
            yield self
        finally:
            _correlation.reset(token)
            _current.reset(bus_token)


_default = DiagnosticBus()
_subscribers = _default._subscribers  # legacy inspection compatibility


def current():
    return _current.get() or _default


def subscribe(fn):
    return current().subscribe(fn)


def unsubscribe(fn):
    current().unsubscribe(fn)


def emit(event):
    current().emit(event)


def active():
    return bool(current()._subscribers)
