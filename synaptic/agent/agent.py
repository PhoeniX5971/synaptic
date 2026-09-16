from ._stream import AgentStream
from datetime import datetime, timezone
from typing import Callable, List, Optional

from ..core.base import ResponseMem, UserMem
from ..core.model import Model
from ..core.base.content import media_content
from ..core.tool import Tool, ToolRegistry
from ..logging import _bus
from ..signal.collector import SignalMode, needs_text_mode
from ..signal.dsl import DSLParser, build_dsl_instructions
from . import _execution
from ..core.invocation import LocalExecutor
from .events import EventBus
from .session import Session, BudgetExceeded
from ..core._ownership import guarded
from ..core._execution_state import check_cancelled
import sys
import asyncio

Permission = Callable[[str, dict], bool]
_PROTOCOL_REPROMPT = (
    "No <tool> or <final> emitted. Emit <tool> to continue or <final></final> to finish."
)


class Agent(AgentStream, _execution.AgentExecution):
    def __init__(
        self,
        model: Model,
        tools: Optional[List[Tool]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        session: Optional[Session] = None,
        max_turns: int = 10,
        permission: Optional[Permission] = None,
        events: Optional[EventBus] = None,
        signals: Optional[EventBus] = None,
        signal_mode: SignalMode = SignalMode.NONE,
        max_protocol_retries: int = 2,
        executor=None,
        policy_required: bool = False,
    ) -> None:
        if getattr(model, "_operation_owner", None) is not None:
            raise RuntimeError("Cannot attach an Agent to an active Model")
        if max_turns < 1 or max_protocol_retries < 0:
            raise ValueError(
                "Positive turn budget and nonnegative protocol retry budget required"
            )
        self.model = model
        self.session = session or Session()
        self.max_turns = max_turns
        self.permission = permission
        self.executor, self.policy_required = executor or LocalExecutor(), policy_required
        self.events = events or EventBus()
        self.signals = signals
        self._registry = tool_registry or ToolRegistry()
        for t in tools or []:
            self._registry.register(t)
        self._base_instructions = model.instructions or ""
        self._text_mode = needs_text_mode(model.provider, signal_mode)
        self._dsl_parser = None
        if self._text_mode:
            self._dsl_parser = DSLParser()
            model.instructions = self._base_instructions + build_dsl_instructions(
                self._registry.all().values()
            )
        else:
            model.bind_tools(list(self._registry.all().values()))
        self._managed_names = set(self._registry.all())
        self.max_protocol_retries = max_protocol_retries
        self.model.history = self.session.history

    def on(self, event: str, fn) -> None:
        return self.events.on(event, fn)

    def register_block(self, name: str, handler=None) -> None:
        if self._dsl_parser is None:
            raise RuntimeError("register_block requires signal mode")
        self._dsl_parser.register(name, handler)

    def _prepare_turn(self):
        self.model.last_response = None
        self._turn_tools = self._registry.all()
        if not self._text_mode:
            if hasattr(self.model, "unbind_tools"):
                self.model.unbind_tools(*self._managed_names)
                self.model.bind_tools(list(self._turn_tools.values()), replace=True)
            self._managed_names = set(self._turn_tools)

    def _start(self) -> None:
        if self.session.state != "idle":
            raise RuntimeError(f"Session is {self.session.state!r}")
        self.session.state = "running"
        self.model.history = self.session.history
        _bus.emit({"type": "agent.start", "session_id": self.session.id})

    def _finish(self) -> None:
        error = sys.exc_info()[1]
        outcome = (
            "cancelled"
            if isinstance(error, (asyncio.CancelledError, GeneratorExit))
            else "exhausted"
            if isinstance(error, BudgetExceeded)
            else "failed"
            if error is not None
            else "completed"
        )
        self.session.outcome = outcome
        if outcome == "completed":
            self.session.state = "done"
        elif isinstance(error, (asyncio.CancelledError, GeneratorExit)):
            self.session.state = "cancelled"
        elif isinstance(error, BudgetExceeded):
            self.session.state = "exhausted"
        else:
            self.session.state = "failed"
        _bus.emit({"type": "agent.end", "session_id": self.session.id})

    def _add_user(self, prompt: Optional[str], role: str, images=None, audio=None) -> None:
        if prompt is not None or images or audio:
            created = datetime.now().astimezone(timezone.utc)
            self.session.history.add(
                UserMem(
                    message=prompt or "",
                    role=role,
                    created=created,
                    content=media_content(images, audio),
                )
            )

    @guarded
    def run(
        self, prompt: Optional[str], role: str = "user", abort=None, **kwargs
    ) -> ResponseMem:
        self._start()
        try:
            next_prompt = None
            self._add_user(prompt, role, kwargs.pop("images", None), kwargs.pop("audio", None))
            for _ in range(self.max_turns):
                check_cancelled(abort)
                self._prepare_turn()
                mem = self.model.invoke(
                    next_prompt, role=role, autorun=False, automem=False, abort=abort, **kwargs
                )
                self.session.history.add(mem)
                self._tool_results(mem.tool_calls, mem, abort)
                if not mem.tool_calls:
                    return mem
                next_prompt = None
                role = "user"
            raise BudgetExceeded("Agent reached max_turns")
        except BaseException:
            self._retain_generation_error()
            raise
        finally:
            self._finish()

    @guarded
    async def arun(
        self, prompt: Optional[str], role: str = "user", abort=None, **kwargs
    ) -> ResponseMem:
        self._start()
        try:
            next_prompt = None
            self._add_user(prompt, role, kwargs.pop("images", None), kwargs.pop("audio", None))
            for _ in range(self.max_turns):
                check_cancelled(abort)
                self._prepare_turn()
                mem = await self.model.ainvoke(
                    next_prompt, role=role, autorun=False, automem=False, abort=abort, **kwargs
                )
                self.session.history.add(mem)
                await self._atool_results(mem.tool_calls, mem, abort)
                if not mem.tool_calls:
                    return mem
                next_prompt = None
                role = "user"
            raise BudgetExceeded("Agent reached max_turns")
        except BaseException:
            self._retain_generation_error()
            raise
        finally:
            self._finish()
