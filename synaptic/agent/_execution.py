"""Prepare before policy; observers only receive detached proposals and results."""

import copy
import inspect

from ..core._awaitables import require_sync
from ..logging import _bus
from ..core.invocation import LocalExecutor
from ..core._execution_state import ExecutionState, admit, check_cancelled
from ..core.base import ResponseMem


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
    check_cancelled(abort)
    invocation = prepare(agent, call)
    if state:
        invocation = state.bind(invocation)
    agent.events.emit("tool_start", copy.deepcopy(call))
    check_cancelled(abort)
    allowed = require_sync(permitted(agent, invocation), "Async permission requires arun()")
    check_cancelled(abort)
    if not allowed:
        return {"name": call.name, "call_id": call.id, "error": "Permission denied"}
    executor = getattr(agent, "executor", LocalExecutor())
    admit(executor, invocation)
    with _bus.current().scope(call_id=invocation.call_id):
        value = require_sync(
            executor.execute(invocation),
            "Async executor requires arun()",
        )
    return {"name": invocation.name, "call_id": invocation.call_id, "result": value}


async def aexecute(agent, call, state=None, abort=None):
    check_cancelled(abort)
    invocation = prepare(agent, call)
    if state:
        invocation = state.bind(invocation)
    await agent.events.aemit("tool_start", copy.deepcopy(call))
    check_cancelled(abort)
    allowed = permitted(agent, invocation)
    if inspect.isawaitable(allowed):
        allowed = await allowed
    check_cancelled(abort)
    if not allowed:
        return {"name": call.name, "call_id": call.id, "error": "Permission denied"}
    executor = getattr(agent, "executor", LocalExecutor())
    admit(executor, invocation)
    with _bus.current().scope(call_id=invocation.call_id):
        value = await executor.aexecute(invocation)
    return {"name": invocation.name, "call_id": invocation.call_id, "result": value}


from typing import List, Any
from ..core.tool import ToolCall
from ..signal.events import ToolCallResult as _SignalTCR


from ._state import AgentState


class AgentExecution(AgentState):
    def _tool_results(self, calls: List[ToolCall], memory=None, abort=None) -> List[Any]:
        mem = memory if memory is not None else ResponseMem("", None, tool_calls=calls)
        with ExecutionState(mem, abort) as state:
            state.check()
            if memory is not None:
                self.events.emit("step_end", mem)
                state.check()
            for call in calls:
                result = execute(self, call, state, abort)
                state.record(call, result)
                self.events.emit("tool_end", copy.deepcopy(call), copy.deepcopy(result))
                self._log_tool(call, result)
                state.check()
        return mem.tool_results

    async def _atool_results(self, calls: List[ToolCall], memory=None, abort=None) -> List[Any]:
        mem = memory if memory is not None else ResponseMem("", None, tool_calls=calls)
        with ExecutionState(mem, abort) as state:
            state.check()
            if memory is not None:
                await self.events.aemit("step_end", mem)
                state.check()
            for call in calls:
                result = await aexecute(self, call, state, abort)
                state.record(call, result)
                await self.events.aemit("tool_end", copy.deepcopy(call), copy.deepcopy(result))
                state.check()
                if self.signals:
                    await self.signals.aemit(
                        "ToolCallResult", _SignalTCR(call=call, result=result)
                    )
                self._log_tool(call, result)
                state.check()
        return mem.tool_results

    def _log_tool(self, call, result):
        _bus.emit(
            {
                "type": "agent.tool",
                "session_id": self.session.id,
                "name": call.name,
                "args": call.args,
                "result": result,
            }
        )
