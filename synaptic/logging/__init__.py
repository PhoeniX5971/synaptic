from typing import Optional

from . import _bus
from ._writer import JSONLWriter

_writer: Optional[JSONLWriter] = None
_subscription = None


def enable_logging(path: str, stream_chunks: bool = False) -> None:
    """Write structured JSONL logs to `path`. Call once at app startup.

    stream_chunks: also log every text delta from astream (verbose; off by default).
    """
    global _writer, _subscription
    disable_logging()
    _bus.stream_chunks = stream_chunks
    _writer = JSONLWriter(path)
    _subscription = _bus.subscribe(_writer.write)


def disable_logging() -> None:
    """Stop logging and close the log file."""
    global _writer, _subscription
    if _writer is not None:
        if _subscription is not None:
            _subscription.close()
        _subscription = None
        _writer.close()
        _writer = None
    _bus.stream_chunks = False


from ._bus import DiagnosticBus
from ._delivery import QueuedSink
