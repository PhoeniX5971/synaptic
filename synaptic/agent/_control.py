"""Run-time control: task state, context compilation, model handoff and delegation."""

from typing import Any, Callable, Dict, List, Optional

from ..core.base import Memory
from ..core.tool import Tool
from ..logging import _bus
from ..signal.collector import SignalMode, needs_text_mode
from ..signal.dsl import build_dsl_instructions
from ._delegate import subagent
from ._state import AgentState

ContextCompiler = Callable[[List[Memory], Dict[str, Any]], List[Memory]]


class AgentControl(AgentState):
    context: Optional[ContextCompiler] = None
    _signal_mode: SignalMode = SignalMode.NONE
    _handoff: Any = None
    _managed_names: set

    @property
    def task(self) -> Dict[str, Any]:
        """The session's canonical task state; tools and compilers share this object."""
        return self.session.task

    def as_tool(self, name: str, description: str, *, sync: bool = False) -> Tool:
        """This agent as a delegation tool for another agent."""
        return subagent(name, description, self, sync=sync)

    def handoff(self, model) -> None:
        """Continue on another model from the next turn boundary, same session and tools.

        Safe to call from an event handler or another task during a run. The response
        being generated finishes on the current model.
        """
        if needs_text_mode(model.provider, self._signal_mode) != self._text_mode:
            raise ValueError("Handoff target must use the same signal mode")
        self._handoff = model

    def _turn_hook(self) -> None:
        """Runs at every turn boundary, before tools are bound for the next model call."""
        target, self._handoff = self._handoff, None
        if target is not None and target is not self.model:
            self._switch(target)
        if self.context is not None:
            compiler, session = self.context, self.session
            session.history.compiler = lambda mems: compiler(mems, session.task)

    def _switch(self, target) -> None:
        if getattr(target, "_operation_owner", None) is not None:
            raise RuntimeError("Cannot hand off to an active Model")
        self._claim(target)
        previous, tools = self.model, list(self._registry.all().values())
        previous._agent_run = None
        if not self._text_mode and hasattr(previous, "unbind_tools"):
            previous.unbind_tools(*self._managed_names)
        self._managed_names = set()
        self._base_instructions = target.instructions or self._base_instructions
        self.model = target
        target.history = self.session.history
        if self._text_mode:
            target.instructions = self._base_instructions + build_dsl_instructions(tools)
        else:
            target.instructions = self._base_instructions
            if not hasattr(target, "unbind_tools"):
                target.bind_tools(tools)
        names = {"source": getattr(previous, "model", None), "target": getattr(target, "model", None)}
        _bus.emit({"type": "agent.handoff", "session_id": self.session.id, **names})
        self._note("handoff", **names)
