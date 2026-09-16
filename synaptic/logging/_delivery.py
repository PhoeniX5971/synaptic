"""Bounded optional asynchronous sink delivery with visible loss/failure counters."""

import queue
import threading
import warnings


class QueuedSink:
    def __init__(self, sink, *, capacity=256, on_error=None):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.sink, self.on_error = sink, on_error
        self.queue = queue.Queue(maxsize=capacity)
        self.failures = self.dropped = 0
        self.closed = False
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="synaptic-log")
        self._worker.start()

    def __call__(self, event):
        if self.closed:
            raise RuntimeError("Sink closed")
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1
            self._report("Diagnostic queue full")

    def _report(self, message):
        try:
            if self.on_error:
                from ..core._awaitables import require_sync

                require_sync(self.on_error(message), "Error handlers must be synchronous")
            else:
                warnings.warn(message, RuntimeWarning)
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set() or not self.queue.empty():
            try:
                event = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                from ..core._awaitables import require_sync

                require_sync(self.sink(event), "QueuedSink requires a synchronous sink")
            except Exception as exc:
                self.failures += 1
                self._report("Diagnostic sink failed: " + type(exc).__name__)
            finally:
                self.queue.task_done()

    def close(self, timeout=5.0):
        self.closed = True
        self._stop.set()
        self._worker.join(timeout)
        if self._worker.is_alive():
            raise TimeoutError("Diagnostic sink did not drain before deadline")
