"""Shared member contract for the Agent runtime mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING
from .events import EventBus
from .session import Session
from ..core.tool import ToolRegistry
from ..signal.dsl import DSLParser

if TYPE_CHECKING:
    from ..core.model import Model


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
    max_protocol_retries: int

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
