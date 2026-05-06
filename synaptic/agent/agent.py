import inspect
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, List, Optional

from ..core._runner import run_tools_async, run_tools_sync
from ..core.base import ResponseChunk, ResponseMem, UserMem, complete_tool_call
from ..core.model import Model
from ..core.tool import ToolCall
from .events import EventBus
from .session import Session

Permission = Callable[[str, dict], bool]


class Agent:
    """Multi-turn agentic loop built on top of `Model`.

    Loops until the model returns no tool calls or `max_turns` is hit.
    Every streaming turn yields chunks character-by-character; tool results
    are written back into history and the next turn continues with prompt=None.
    """

    def __init__(
        self,
        model: Model,
        session: Optional[Session] = None,
        max_turns: int = 10,
        permission: Optional[Permission] = None,
        events: Optional[EventBus] = None,
    ) -> None:
        self.model = model
        self.session = session or Session()
        self.max_turns = max_turns
        self.permission = permission
        self.events = events or EventBus()
        self.model.history = self.session.history

    def on(self, event: str, fn) -> None:
        """Subscribe to `text_delta`, `tool_start`, `tool_end`, or `step_end`."""
        self.events.on(event, fn)

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
                result = run_tools_sync(self.model.llm.synaptic_tools, self.model.blacklist, [call])[0]
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
                result = (await run_tools_async(self.model.llm.synaptic_tools, self.model.blacklist, [call]))[0]
            await self.events.aemit("tool_end", call, result)
            results.append(result)
        return results

    def run(self, prompt: Optional[str], role: str = "user", **kwargs) -> ResponseMem:
        """Run the agent loop synchronously until completion."""
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

                async for chunk in self.model.llm.astream(
                    prompt=next_prompt, role=role, abort=abort, **kwargs
                ):
                    if abort and abort.is_set():
                        return
                    # Accumulate non-final text deltas; skip the final summary chunk
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
