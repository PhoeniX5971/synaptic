import inspect
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, List, Optional

from ..core._runner import run_tools_async, run_tools_sync
from ..core.base import ResponseChunk, ResponseMem, UserMem, complete_tool_call
from ..core.model import Model
from ..core.tool import Tool, ToolCall, ToolRegistry
from ..signal.collector import SignalMode, collect as _signal_collect, needs_text_mode
from ..signal.dsl import DSLParser, build_dsl_instructions
from ..signal.events import ToolCallResult as _SignalTCR
from .events import EventBus
from .session import Session

Permission = Callable[[str, dict], bool]


class Agent:
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
    ) -> None:
        self.model = model
        self.session = session or Session()
        self.max_turns = max_turns
        self.permission = permission
        self.events = events or EventBus()
        self.signals = signals
        self._registry = tool_registry or ToolRegistry()
        for t in (tools or []):
            self._registry.register(t)
        self._base_instructions = model.instructions or ""
        self._text_mode = needs_text_mode(model.provider, signal_mode)
        self._dsl_parser = None
        if self._text_mode:
            self._dsl_parser = DSLParser()
            model.instructions = self._base_instructions + build_dsl_instructions(self._registry.all().values())
        else:
            model.bind_tools(list(self._registry.all().values()))
        self.model.history = self.session.history

    def on(self, event: str, fn) -> None:
        self.events.on(event, fn)

    def register_block(self, name: str, handler=None) -> None:
        if self._dsl_parser is None:
            raise RuntimeError("register_block requires signal mode")
        self._dsl_parser.register(name, handler)

    def _start(self) -> None:
        if self.session.state == "running":
            raise RuntimeError("Session is already running")
        self.session.state = "running"
        self.model.history = self.session.history

    def _finish(self) -> None:
        self.session.state = "done"

    def _add_user(self, prompt: Optional[str], role: str) -> None:
        if prompt is not None:
            created = datetime.now().astimezone(timezone.utc)
            self.session.history.add(UserMem(message=prompt, role=role, created=created))

    def _allow(self, call: ToolCall) -> bool:
        if self.permission is None:
            return True
        allowed = self.permission(call.name, call.args)
        if inspect.isawaitable(allowed):
            raise RuntimeError("Async permission requires arun()")
        return bool(allowed)

    async def _aallow(self, call: ToolCall) -> bool:
        if self.permission is None:
            return True
        allowed = self.permission(call.name, call.args)
        if inspect.isawaitable(allowed):
            allowed = await allowed
        return bool(allowed)

    def _tool_results(self, calls: List[ToolCall]) -> List[Any]:
        results: List[Any] = []
        for call in calls:
            self.events.emit("tool_start", call)
            if not self._allow(call):
                result = {"name": call.name, "error": "Permission denied"}
            else:
                result = run_tools_sync(list(self._registry.all().values()), self.model.blacklist, [call])[0]
            self.events.emit("tool_end", call, result)
            results.append(result)
        return results

    async def _atool_results(self, calls: List[ToolCall]) -> List[Any]:
        results: List[Any] = []
        for call in calls:
            await self.events.aemit("tool_start", call)
            if not await self._aallow(call):
                result = {"name": call.name, "error": "Permission denied"}
            else:
                result = (await run_tools_async(list(self._registry.all().values()), self.model.blacklist, [call]))[0]
            await self.events.aemit("tool_end", call, result)
            if self.signals:
                await self.signals.aemit("ToolCallResult", _SignalTCR(call=call, result=result))
            results.append(result)
        return results

    def run(self, prompt: Optional[str], role: str = "user", **kwargs) -> ResponseMem:
        self._start()
        try:
            next_prompt = prompt
            self._add_user(prompt, role)
            for _ in range(self.max_turns):
                mem = self.model.invoke(next_prompt, role=role, autorun=False, automem=False, **kwargs)
                self.session.history.add(mem)
                self.events.emit("step_end", mem)
                if not mem.tool_calls:
                    return mem
                complete_tool_call(mem, self._tool_results(mem.tool_calls), self.session.history)
                next_prompt = None
                role = "user"
            raise RuntimeError("Agent reached max_turns")
        finally:
            self._finish()

    async def arun(self, prompt: Optional[str], role: str = "user", **kwargs) -> ResponseMem:
        """Run the agent loop from async code until completion."""
        self._start()
        try:
            next_prompt = prompt
            self._add_user(prompt, role)
            for _ in range(self.max_turns):
                mem = await self.model.ainvoke(next_prompt, role=role, autorun=False, automem=False, **kwargs)
                self.session.history.add(mem)
                await self.events.aemit("step_end", mem)
                if not mem.tool_calls:
                    return mem
                complete_tool_call(mem, await self._atool_results(mem.tool_calls), self.session.history)
                next_prompt = None
                role = "user"
            raise RuntimeError("Agent reached max_turns")
        finally:
            self._finish()

    async def astream(
        self, prompt: Optional[str], role: str = "user", abort=None, **kwargs
    ) -> AsyncIterator[ResponseChunk]:
        """Full multi-turn streaming loop.

        Each turn streams text character-by-character. After a turn ends:
        - tool calls are executed (with permission checks + events)
        - results are written into history
        - the next turn begins immediately with prompt=None
        Loops until no tool calls remain or max_turns is hit.
        """
        self._start()
        try:
            next_prompt = prompt
            self._add_user(prompt, role)
            for _ in range(self.max_turns):
                accumulated = ""
                tool_calls: List[ToolCall] = []
                created = datetime.now().astimezone(timezone.utc)

                if self._text_mode:
                    self.model.instructions = self._base_instructions + build_dsl_instructions(self._registry.all().values())
                _s = self.model.llm.astream(prompt=next_prompt, role=role, abort=abort, **kwargs)
                if self.signals:
                    _s = _signal_collect(_s, self.signals, self._text_mode, self._dsl_parser)
                async for chunk in _s:
                    if abort and abort.is_set():
                        return
                    if not chunk.is_final and chunk.text:
                        accumulated += chunk.text
                        await self.events.aemit("text_delta", chunk.text)
                    if chunk.function_call:
                        tool_calls.append(chunk.function_call)
                    yield chunk

                mem = ResponseMem(message=accumulated, created=created, tool_calls=tool_calls)
                self.session.history.add(mem)
                await self.events.aemit("step_end", mem)

                if not tool_calls:
                    return

                results = await self._atool_results(tool_calls)
                complete_tool_call(mem, results, self.session.history)
                next_prompt = None
                role = "user"

            raise RuntimeError("Agent reached max_turns")
        finally:
            self._finish()
