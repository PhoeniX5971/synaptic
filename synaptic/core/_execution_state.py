"""Incremental receipts and conservative effect state, persisted in history extensions."""

import asyncio
import copy
import threading
from dataclasses import replace


def check_cancelled(abort=None):
    try:
        task = asyncio.current_task()
    except RuntimeError:  # Synchronous callers and scheduler worker threads.
        task = None
    if (abort is not None and abort.is_set()) or (task and task.cancelling()):
        raise asyncio.CancelledError()


class ExecutionState:
    """One response's dispatch journal. Interrupted responses require reconciliation."""

    def __init__(self, memory, abort=None):
        self.memory, self.abort = memory, abort
        self._lock = threading.RLock()
        self._interrupted = False
        self.calls = memory.tool_outcomes
        ids = [call.id for call in memory.tool_calls]
        if len(ids) != len(set(ids)):
            memory.extensions["incomplete"] = True
            raise ValueError("Duplicate tool call IDs cannot be dispatched")
        for call in memory.tool_calls:
            self.calls.setdefault(call.id, {"name": call.name, "status": "not_started"})
        for index, result in enumerate(memory.tool_results):
            call_id = result.get("call_id") if isinstance(result, dict) else None
            if call_id is None and len(memory.tool_results) == len(ids):
                call_id = ids[index]
            if call_id in self.calls and self.calls[call_id]["status"] == "not_started":
                self.calls[call_id]["status"] = "reconciled"
        self._publish()

    def _publish(self):
        if self.calls:
            self.memory.extensions["execution"] = {
                "version": 1,
                "calls": copy.deepcopy(self.calls),
            }

    def __enter__(self):
        self.memory.extensions["incomplete"] = True
        return self

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            self.interrupt(error)
        else:
            self.memory.extensions.pop("incomplete", None)
        return False

    def interrupt(self, error):
        with self._lock:
            self._interrupted = True
            self.calls = self.memory.tool_outcomes
            for outcome in self.calls.values():
                if outcome["status"] == "started":
                    outcome["status"] = "uncertain"
            self.memory.extensions["incomplete"] = True
            self.memory.extensions["execution_error"] = type(error).__name__
            self._publish()
            self.memory.enforce_budgets()

    def check(self):
        check_cancelled(self.abort)
        if self._interrupted:
            raise asyncio.CancelledError()

    def bind(self, invocation):
        self.check()
        status = self.calls[invocation.call_id]["status"]
        if status != "not_started":
            raise RuntimeError(
                f"Call {invocation.call_id} is {status}; execution cannot be replayed"
            )
        return replace(invocation, _dispatch=Dispatch(self, invocation.call_id))

    def record(self, call, result):
        with self._lock:
            # Persist before observers or another call can fail.
            self.memory.tool_results.append(copy.deepcopy(result))
            self.calls[call.id]["status"] = "succeeded" if "result" in result else "denied"
            self._publish()
            self.memory.enforce_budgets()


class Dispatch:
    def __init__(self, state, call_id):
        self.state, self.call_id = state, call_id
        self._begun = False

    def check(self):
        self.state.check()

    def begin(self, opaque=False):
        with self.state._lock:
            self.check()
            outcome = self.state.calls[self.call_id]
            if self._begun or outcome["status"] not in ("not_started", "started"):
                raise RuntimeError("An invocation may begin its effect only once")
            self._begun = not opaque
            outcome["status"] = "started"
            self.state._publish()


def admit(executor, invocation):
    """Opaque executors begin here; cooperative executors gate their own admission."""
    invocation.check_cancelled()
    if not getattr(executor, "manages_admission", False):
        if invocation._dispatch:
            invocation._dispatch.begin(opaque=True)
