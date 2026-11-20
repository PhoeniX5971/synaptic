"""Append-only ordered record of agent runs for inspection, forking and replay."""

import copy
import threading
import warnings
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from ..core._registry import Subscription

Entry = dict[str, Any]


class Ledger:
    """What a run did, in order. Entries are never edited or removed.

    `sink` receives a copy of every entry as it is appended; it is the consumer's
    hook for durable storage. A failing sink warns and never interrupts the run.
    """

    def __init__(
        self,
        entries: Optional[Iterable[Entry]] = None,
        sink: Optional[Callable[[Entry], Any]] = None,
    ):
        self._entries: list[Entry] = [copy.deepcopy(entry) for entry in entries or []]
        self.sink = sink
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[Entry]:
        return copy.deepcopy(self._entries)

    def append(self, kind: str, **data: Any) -> Entry:
        with self._lock:
            entry = {
                "seq": len(self._entries),
                "at": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                **copy.deepcopy(data),
            }
            self._entries.append(entry)
        if self.sink is not None:
            try:
                self.sink(copy.deepcopy(entry))
            except Exception as error:
                warnings.warn("Ledger sink failed: " + type(error).__name__, RuntimeWarning)
        return copy.deepcopy(entry)

    def of(self, *kinds: str) -> list[Entry]:
        return [copy.deepcopy(entry) for entry in self._entries if entry["kind"] in kinds]

    def fork(self, seq: Optional[int] = None) -> "Ledger":
        """Independent copy of the record up to and including `seq` (all when None)."""
        end = len(self._entries) if seq is None else seq + 1
        return Ledger(self._entries[:end])

    def to_list(self) -> list[Entry]:
        return self.entries

    @classmethod
    def from_list(cls, entries: Iterable[Entry], sink=None) -> "Ledger":
        return cls(entries, sink)

    def summary(self) -> dict[str, Any]:
        models, tools = self.of("model"), self.of("tool_end")
        return {
            "runs": len(self.of("run_start")),
            "outcomes": [entry["outcome"] for entry in self.of("run_end")],
            "model_calls": len(models),
            "tool_calls": [entry["name"] for entry in tools],
            "tool_errors": sum("error" in entry for entry in tools),
            "input_tokens": sum(entry.get("input_tokens") or 0 for entry in models),
            "output_tokens": sum(entry.get("output_tokens") or 0 for entry in models),
            "model_ms": sum(entry.get("latency_ms") or 0 for entry in models),
            "tool_ms": sum(entry.get("latency_ms") or 0 for entry in tools),
        }

    def attach(self, agent) -> Subscription:
        """Record an agent's runs; close the returned subscription to stop."""

        def note(kind, **data):
            self.append(kind, session_id=agent.session.id, **data)

        def model(memory):
            note(
                "model",
                provider=str(getattr(agent.model, "provider", "")),
                model=getattr(agent.model, "model", None),
                message=memory.message,
                tool_calls=[
                    {"id": call.id, "name": call.name, "args": call.args}
                    for call in memory.tool_calls
                ],
                input_tokens=memory.input_tokens,
                output_tokens=memory.output_tokens,
                finish_reason=memory.finish_reason,
                latency_ms=memory.latency_ms,
            )

        def started(call):
            note("tool_start", call_id=call.id, name=call.name, args=call.args)

        def ended(call, result):
            body = {key: result[key] for key in ("result", "error", "latency_ms") if key in result}
            note("tool_end", call_id=call.id, name=call.name, **body)

        def failed(call, result):
            note("tool_error", call_id=call.id, name=call.name, **{
                key: result[key] for key in ("error", "latency_ms") if key in result
            })

        previous = agent.ledger
        agent.ledger = self
        subscriptions = [
            agent.events.on("step_end", model),
            agent.events.on("tool_start", started),
            agent.events.on("tool_end", ended),
            agent.events.on("tool_error", failed),
        ]

        def detach():
            for subscription in subscriptions:
                subscription.close()
            if agent.ledger is self:
                agent.ledger = previous

        return Subscription(detach)
