"""Streaming driver for the agent loop; `arun` is this stream, drained."""

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from ..core._completion import Completion
from ..core._execution_state import ExecutionState
from ..core._ownership import guarded
from ..core._runner import astream as model_stream
from ..core.base import ResponseChunk, ResponseMem
from ..signal.collector import collect as _signal_collect
from ..signal.dsl import build_dsl_instructions
from ._control import AgentControl
from ._loop import turns


class AgentStream(AgentControl):
    @guarded
    async def astream(
        self, prompt: Optional[str], role: str = "user", abort=None, **kwargs
    ) -> AsyncIterator[ResponseChunk]:
        self._start()
        drive = self._drive(prompt, role, abort, [], kwargs)
        try:
            async for chunk in drive:
                yield chunk
        finally:
            try:
                await drive.aclose()  # An early consumer close must reach the provider stream.
            finally:
                self._finish()

    @guarded
    async def arun(
        self, prompt: Optional[str], role: str = "user", abort=None, **kwargs
    ) -> ResponseMem:
        self._start()
        try:
            done: list = []
            async for _ in self._drive(prompt, role, abort, done, kwargs):
                pass
            return done[0]
        finally:
            self._finish()

    async def _drive(self, prompt, role, abort, done, kwargs):
        """Step the loop, streaming each model call and awaiting each tool batch."""
        loop = turns(self, prompt, role, kwargs.pop("images", None), kwargs.pop("audio", None), abort)
        try:
            step, reply = next(loop), None
            while True:
                if step[0] == "model":
                    holder: list = []
                    model_step = self._model_step(step[1], abort, holder, kwargs)
                    try:
                        async for chunk in model_step:
                            yield chunk
                    finally:
                        await model_step.aclose()
                    reply = holder[0]
                else:
                    await self._atool_results(step[1].tool_calls, step[1], abort)
                    reply = None
                    while self._outbox:
                        event, args = self._outbox.pop(0)
                        await self.events.aemit(event, *args)
                step = loop.send(reply)
        except StopIteration as finished:
            done.append(result := finished.value)
            if result.output_name:
                await self.events.aemit("final", result.output_name, result.output)
                extra = {"output_name": result.output_name, "output": result.output}
                yield ResponseChunk("", is_final=True, extensions=extra)
        finally:
            loop.close()

    async def _model_step(self, role, abort, holder, kwargs):
        completion, final_seen = Completion(), False
        created = datetime.now().astimezone(timezone.utc)
        if self._text_mode:
            self.model.instructions = self._compose() + build_dsl_instructions(
                self._turn_tools.values()
            )
        source = model_stream(
            self.model, prompt=None, role=role, abort=abort, autorun=False, automem=False, **kwargs
        )
        if self.signals:
            source = _signal_collect(source, self.signals, self._text_mode, self._dsl_parser)
        try:
            async for chunk in source:
                if abort and abort.is_set():
                    raise asyncio.CancelledError()
                completion.add(chunk)
                final_seen = final_seen or bool(chunk.final)
                if not chunk.is_final and chunk.text:
                    await self.events.aemit("text_delta", chunk.text)
                yield chunk
        except BaseException as error:
            partial = completion.memory(created)
            ExecutionState(partial, abort).interrupt(error)
            self.session.history.add(partial)
            raise
        finally:
            close = getattr(source, "aclose", None)
            if close:
                await close()
        memory = completion.memory(created)
        memory.latency_ms = getattr(self.model.last_response, "latency_ms", None)
        holder.append((memory, final_seen))
