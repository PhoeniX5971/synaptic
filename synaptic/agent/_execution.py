"""Prepare before policy; observers only receive detached proposals and results."""

import copy
import inspect

from ..core._awaitables import require_sync
from ..logging import _bus
from ..core.invocation import LocalExecutor


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


def execute(agent, call):
    invocation = prepare(agent, call)
    agent.events.emit("tool_start", copy.deepcopy(call))
    allowed = require_sync(permitted(agent, invocation), "Async permission requires arun()")
    if not allowed:
        return {"name": call.name, "call_id": call.id, "error": "Permission denied"}
    with _bus.current().scope(call_id=invocation.call_id):
        value = require_sync(
            getattr(agent, "executor", LocalExecutor()).execute(invocation),
            "Async executor requires arun()",
        )
    return {"name": invocation.name, "call_id": invocation.call_id, "result": value}


async def aexecute(agent, call):
    invocation = prepare(agent, call)
    await agent.events.aemit("tool_start", copy.deepcopy(call))
    allowed = permitted(agent, invocation)
    if inspect.isawaitable(allowed):
        allowed = await allowed
    if not allowed:
        return {"name": call.name, "call_id": call.id, "error": "Permission denied"}
    with _bus.current().scope(call_id=invocation.call_id):
        value = await getattr(agent, "executor", LocalExecutor()).aexecute(invocation)
    return {"name": invocation.name, "call_id": invocation.call_id, "result": value}


from typing import List, Any
from ..core.tool import ToolCall
from ..signal.events import ToolCallResult as _SignalTCR


class AgentExecution:
    def _tool_results(self, calls: List[ToolCall]) -> List[Any]:
        results: List[Any] = []
        for call in calls:
            result = execute(self, call)
            self.events.emit("tool_end", copy.deepcopy(call), copy.deepcopy(result))
            _bus.emit(
                {
                    "type": "agent.tool",
                    "session_id": self.session.id,
                    "name": call.name,
                    "args": call.args,
                    "result": result,
                }
            )
            results.append(result)
        return results

    async def _atool_results(self, calls: List[ToolCall]) -> List[Any]:
        results: List[Any] = []
        for call in calls:
            result = await aexecute(self, call)
            await self.events.aemit("tool_end", copy.deepcopy(call), copy.deepcopy(result))
            if self.signals:
                await self.signals.aemit("ToolCallResult", _SignalTCR(call=call, result=result))
            _bus.emit(
                {
                    "type": "agent.tool",
                    "session_id": self.session.id,
                    "name": call.name,
                    "args": call.args,
                    "result": result,
                }
            )
            results.append(result)
        return results
