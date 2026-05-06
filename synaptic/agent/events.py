import inspect
from typing import Any, Callable, Dict, List


class EventBus:
    """Small event dispatcher used by agents and streaming integrations."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable[..., Any]]] = {}

    def on(self, event: str, fn: Callable[..., Any]) -> None:
        """Register a sync or async handler for an event name."""
        self._handlers.setdefault(event, []).append(fn)

    def emit(self, event: str, *args: Any) -> None:
        """Emit an event to sync handlers."""
        for fn in self._handlers.get(event, []):
            result = fn(*args)
            if inspect.isawaitable(result):
                raise RuntimeError("Async event handlers require aemit()")

    async def aemit(self, event: str, *args: Any) -> None:
        """Emit an event and await handlers that return awaitables."""
        for fn in self._handlers.get(event, []):
            result = fn(*args)
            if inspect.isawaitable(result):
                await result
