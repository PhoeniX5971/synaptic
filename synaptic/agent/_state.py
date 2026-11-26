"""Shared member contract for the Agent runtime mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING
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
