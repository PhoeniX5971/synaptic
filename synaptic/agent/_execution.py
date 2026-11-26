"""Prepare before policy; observers only receive detached proposals and results."""

import copy
import inspect
import time
from typing import Any, List

from ..core._awaitables import require_sync
from ..logging import _bus
from ..core.invocation import LocalExecutor
from ..core._execution_state import ExecutionState, admit, check_cancelled
from ..core.base import ResponseMem
from ..core.tool import ToolCall
from ..signal.events import ToolCallResult as _SignalTCR
from ._delegate import abort_scope
from ._parallel import batches, gather
from ._state import AgentState


def prepare(agent, call):
    if call.name in agent.model.blacklist:
        raise PermissionError("Tool is blacklisted")
    tool = getattr(agent, "_turn_tools", agent._registry.all()).get(call.name)
    if tool is None:
        raise ValueError("Tool not registered")
    return tool.prepare(call)


def permitted(agent, invocation):
    policy = agent.permission
    if policy is None:
        if getattr(agent, "policy_required", False):
            raise PermissionError("An execution policy is required")
        return True
    return policy(invocation.name, invocation.arguments)


def execute(agent, call, state=None, abort=None):
    return _timed(agent, call, state, abort)[0]


def _failed(agent, call, error):
    """Error receipt and whether this tool's failures go back to the model."""
    tool = (getattr(agent, "_turn_tools", None) or agent._registry.all()).get(call.name)
    mode = (tool.metadata.get("errors") if tool else None) or getattr(agent, "tool_errors", "raise")
    receipt = {"name": call.name, "call_id": call.id, "error": f"{type(error).__name__}: {error}"}
    return receipt, mode == "return"


def _ms(start):
    return int((time.monotonic() - start) * 1000)


def _observed(result, latency_ms):
    """Observer copy of a receipt; timing never enters history or provider requests."""
    return {**copy.deepcopy(result), "latency_ms": latency_ms}


def _timed(agent, call, state=None, abort=None):
    """Return the receipt and executor-only latency; policy and observer time is excluded."""
    check_cancelled(abort)
    try:
        invocation = prepare(agent, call)
    except Exception as error:  # Unknown tool or invalid arguments: no effect started.
        receipt, returned = _failed(agent, call, error)
        if not returned:
            raise
        return receipt, 0, "failed"
    if state:
        invocation = state.bind(invocation)
    agent.events.emit("tool_start", copy.deepcopy(call))
    check_cancelled(abort)
    allowed = require_sync(permitted(agent, invocation), "Async permission requires arun()")
    check_cancelled(abort)
    if not allowed:
        return {"name": call.name, "call_id": call.id, "error": "Permission denied"}, 0, "denied"
    executor = getattr(agent, "executor", LocalExecutor())
    admit(executor, invocation)
    start = time.monotonic()
    try:
        with _bus.current().scope(call_id=invocation.call_id):
            value = require_sync(executor.execute(invocation), "Async executor requires arun()")
    except Exception as error:
        receipt, returned = _failed(agent, call, error)
        if returned:
            return receipt, _ms(start), "failed"
        agent.events.emit("tool_error", copy.deepcopy(call), _observed(receipt, _ms(start)))
        raise
    receipt = {"name": invocation.name, "call_id": invocation.call_id, "result": value}
    return receipt, _ms(start), "succeeded"


async def aexecute(agent, call, state=None, abort=None):
    return (await _atimed(agent, call, state, abort))[0]


async def _atimed(agent, call, state=None, abort=None):
    check_cancelled(abort)
    try:
        invocation = prepare(agent, call)
    except Exception as error:
        receipt, returned = _failed(agent, call, error)
        if not returned:
            raise
        return receipt, 0, "failed"
    if state:
        invocation = state.bind(invocation)
    await agent.events.aemit("tool_start", copy.deepcopy(call))
    check_cancelled(abort)
    allowed = permitted(agent, invocation)
    if inspect.isawaitable(allowed):
        allowed = await allowed
    check_cancelled(abort)
    if not allowed:
        return {"name": call.name, "call_id": call.id, "error": "Permission denied"}, 0, "denied"
    executor = getattr(agent, "executor", LocalExecutor())
    admit(executor, invocation)
    start = time.monotonic()
    try:
        with _bus.current().scope(call_id=invocation.call_id):
            value = await executor.aexecute(invocation)
    except Exception as error:
        receipt, returned = _failed(agent, call, error)
        if returned:
            return receipt, _ms(start), "failed"
        await agent.events.aemit("tool_error", copy.deepcopy(call), _observed(receipt, _ms(start)))
        raise
    receipt = {"name": invocation.name, "call_id": invocation.call_id, "result": value}
    return receipt, _ms(start), "succeeded"


def _signal(call, result, ms):
    return _SignalTCR(call=call, result=result, error=result.get("error"), latency_ms=ms)


class AgentExecution(AgentState):
    def _tool_results(self, calls: List[ToolCall], memory=None, abort=None) -> List[Any]:
        mem = memory if memory is not None else ResponseMem("", None, tool_calls=calls)
        with abort_scope(abort), ExecutionState(mem, abort) as state:
            state.check()
            if memory is not None:
                self.events.emit("step_end", mem)
                state.check()
            for call in calls:
                result, ms, status = _timed(self, call, state, abort)
                state.record(call, result, ms, status)
                self.events.emit("tool_end", copy.deepcopy(call), _observed(result, ms))
                state.check()
                if self.signals:
                    self.signals.emit("ToolCallResult", _signal(call, result, ms))
                self._log_tool(call, result, ms)
                state.check()
        return mem.tool_results

    async def _atool_results(self, calls: List[ToolCall], memory=None, abort=None) -> List[Any]:
        mem = memory if memory is not None else ResponseMem("", None, tool_calls=calls)
        with abort_scope(abort), ExecutionState(mem, abort) as state:
            state.check()
            if memory is not None:
                await self.events.aemit("step_end", mem)
                state.check()
            for batch in batches(self, calls):
                outcomes = await gather(self, batch, lambda c: _atimed(self, c, state, abort))
                # Receipts are published in call order before any observer or failure.
                done = [(c, o) for c, o in zip(batch, outcomes) if not isinstance(o, BaseException)]
                for call, (result, ms, status) in done:
                    state.record(call, result, ms, status)
                for outcome in outcomes:
                    if isinstance(outcome, BaseException):
                        raise outcome
                for call, (result, ms, _) in done:
                    await self.events.aemit(
                        "tool_end", copy.deepcopy(call), _observed(result, ms)
                    )
                    state.check()
                    if self.signals:
                        await self.signals.aemit("ToolCallResult", _signal(call, result, ms))
                    self._log_tool(call, result, ms)
                    state.check()
        return mem.tool_results

    def _log_tool(self, call, result, latency_ms):
        _bus.emit(
            {
                "type": "agent.tool",
                "session_id": self.session.id,
                "name": call.name,
                "args": call.args,
                "result": result,
                "latency_ms": latency_ms,
            }
        )
