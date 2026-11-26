"""Shared member contract for the Agent runtime mixins."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..logging import _bus
from .events import EventBus
from .session import Session
from ..core.tool import ToolRegistry
from ..signal.dsl import DSLParser

if TYPE_CHECKING:
    from ..core.model import Model


def media_refs(images, audio) -> dict:
    """String media references (paths, URLs) are recordable; in-memory objects are not."""
    refs = {
        "images": [item for item in images or [] if isinstance(item, str)],
        "audio": [item for item in audio or [] if isinstance(item, str)],
    }
    return {key: value for key, value in refs.items() if value}


class AgentState:
    model: Model
    session: Session
    events: EventBus
    signals: EventBus | None
    _registry: ToolRegistry
    _text_mode: bool
    _base_instructions: str
    _dsl_parser: DSLParser | None
    max_turns: int
    _turn_tools: dict
    max_parallel_tools: int = 1
    max_protocol_retries: int

    ledger = None  # Optional run ledger; see synaptic.run.Ledger.attach.
    tool_errors: str = "raise"
    retry = None  # Optional RetryPolicy for model calls that fail before any output.

    def _retry_pause(self, error, emitted: bool, attempt: int, started: float):
        """Seconds to wait before repeating a model call, or None when it must not repeat."""
        policy = self.retry
        if policy is None or emitted or not isinstance(error, Exception):
            return None
        pause = policy.pause(attempt, error, policy.deadline - (time.monotonic() - started))
        if pause is not None:
            detail = {"attempt": attempt + 1, "error": type(error).__name__}
            _bus.emit({"type": "agent.retry", "session_id": self.session.id, **detail})
            self._note("retry", **detail)
        return pause

    def _invoke(self, role, abort, kwargs):
        """Synchronous model step; nothing is visible until it returns, so it may repeat."""
        started, attempt = time.monotonic(), 0
        while True:
            try:
                return self.model.invoke(
                    None, role=role, autorun=False, automem=False, abort=abort, **kwargs
                )
            except Exception as error:
                pause = self._retry_pause(error, False, attempt, started)
                if pause is None:
                    raise
                time.sleep(pause)
                attempt += 1

    def _claim(self, model) -> None:
        """One agent run per Model: a nested run on the same Model would swap its history."""
        owner = getattr(model, "_agent_run", None)
        if owner is not None and owner is not self:
            raise RuntimeError("Model is driving another agent run; give each agent its own Model")
        model._agent_run = self

    def _note(self, kind: str, **data) -> None:
        if self.ledger is not None:
            self.ledger.append(kind, session_id=self.session.id, **data)

    def _retain_generation_error(self):
        partial = getattr(self.model, "last_response", None)
        if partial is not None and partial not in self.session.history.entries:
            self.session.history.add(partial)

    def _start(self) -> None:
        raise NotImplementedError

    def _finish(self) -> None:
        raise NotImplementedError

    def _prepare_turn(self):
        raise NotImplementedError

    def _add_user(self, prompt, role, images=None, audio=None) -> None:
        raise NotImplementedError

    async def _atool_results(self, calls, memory=None, abort=None):
        raise NotImplementedError
