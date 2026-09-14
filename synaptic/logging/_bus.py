from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

_subscribers: List[Callable[[Dict[str, Any]], None]] = []
stream_chunks: bool = False


def subscribe(fn: Callable[[Dict[str, Any]], None]) -> None:
    if fn not in _subscribers:
        _subscribers.append(fn)


def unsubscribe(fn: Callable[[Dict[str, Any]], None]) -> None:
    if fn in _subscribers:
        _subscribers.remove(fn)


def emit(event: Dict[str, Any]) -> None:
    if not _subscribers:
        return
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    for fn in _subscribers:
        try:
            fn(event)
        except Exception:
            pass


def active() -> bool:
    return bool(_subscribers)
