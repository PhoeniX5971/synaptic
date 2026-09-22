"""AgentStream implementation separated by runtime responsibility."""

from datetime import datetime, timezone
from typing import AsyncIterator, Optional
from ..core._completion import Completion
from ..core._runner import astream as model_stream
from ..core.base import ResponseChunk
from ..signal.collector import collect as _signal_collect
from ..signal.dsl import build_dsl_instructions
from .session import BudgetExceeded
from ..core._ownership import guarded
from ..core._execution_state import ExecutionState, check_cancelled
import asyncio

_PROTOCOL_REPROMPT = (
    "No <tool> or <final> emitted. Emit <tool> to continue or <final></final> to finish."
)


from ._state import AgentState


class AgentStream(AgentState):
    @guarded
    async def astream(
        self, prompt: Optional[str], role: str = "user", abort=None, **kwargs
    ) -> AsyncIterator[ResponseChunk]:
        self._start()
        try:
            next_prompt = None
            self._add_user(prompt, role, kwargs.pop("images", None), kwargs.pop("audio", None))
            protocol_retries = 0
            for _ in range(self.max_turns):
                check_cancelled(abort)
                self._prepare_turn()
                completion = Completion()
                final_seen = False
                created = datetime.now().astimezone(timezone.utc)
                if self._text_mode:
                    self.model.instructions = self._base_instructions + build_dsl_instructions(
                        self._registry.all().values()
                    )
                _s = model_stream(
                    self.model,
                    prompt=next_prompt,
                    role=role,
                    abort=abort,
                    autorun=False,
                    automem=False,
                    **kwargs,
                )
                if self.signals:
                    _s = _signal_collect(_s, self.signals, self._text_mode, self._dsl_parser)
                try:
                    async for chunk in _s:
                        if abort and abort.is_set():
                            raise asyncio.CancelledError()
                        completion.add(chunk)
                        if chunk.final:
                            final_seen = True
                        if not chunk.is_final and chunk.text:
                            await self.events.aemit("text_delta", chunk.text)
                        yield chunk
                except BaseException as error:
                    partial = completion.memory(created)
                    ExecutionState(partial, abort).interrupt(error)
                    self.session.history.add(partial)
                    raise
                finally:
                    close = getattr(_s, "aclose", None)
                    if close:
                        await close()
                mem = completion.memory(created)
                tool_calls = mem.tool_calls
                self.session.history.add(mem)
                await self._atool_results(tool_calls, mem, abort)
                if tool_calls:
                    protocol_retries = 0
                    next_prompt = None
                    role = "user"
                    continue
                if not self._text_mode or not self.signals or final_seen:
                    return
                if protocol_retries >= self.max_protocol_retries:
                    raise BudgetExceeded("Agent reached max_protocol_retries")
                protocol_retries += 1
                self._add_user(_PROTOCOL_REPROMPT, "user")
                next_prompt = None
                role = "user"

            raise BudgetExceeded("Agent reached max_turns")
        finally:
            self._finish()
