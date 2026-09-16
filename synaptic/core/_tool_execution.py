"""One-shot dispatch, publishing every receipt before admitting another effect."""

from .tool import ToolCall
from .invocation import LocalExecutor
from .base.memory import ResponseMem
from ._execution_state import ExecutionState, admit


def _prepare(tools, blacklist, call, state):
    state.check()
    if not isinstance(call, ToolCall):
        raise ValueError("Invalid tool call format")
    tool = next((tool for tool in tools if tool.name == call.name), None)
    if tool is None or call.name in blacklist:
        return None
    return state.bind(tool.prepare(call))


def run_tools_sync(tools, blacklist, tool_calls, executor=None, memory=None, abort=None):
    legacy = memory is None
    mem = memory if memory is not None else ResponseMem("", None, tool_calls=tool_calls)
    executor = executor or LocalExecutor()
    with ExecutionState(mem, abort) as state:
        state.check()
        for call in tool_calls:
            result = {"name": call.name, "call_id": call.id}
            try:
                invocation = _prepare(tools, blacklist, call, state)
                if invocation is None:
                    result["error"] = "Tool not registered or blacklisted"
                else:
                    admit(executor, invocation)
                    result["result"] = executor.execute(invocation)
            except Exception as error:
                if not legacy:
                    raise
                result["error"] = str(error)
            state.record(call, result)
            state.check()
    return (
        [{k: v for k, v in item.items() if k != "call_id"} for item in mem.tool_results]
        if legacy
        else mem.tool_results
    )


async def run_tools_async(tools, blacklist, tool_calls, executor=None, memory=None, abort=None):
    legacy = memory is None
    mem = memory if memory is not None else ResponseMem("", None, tool_calls=tool_calls)
    executor = executor or LocalExecutor()
    with ExecutionState(mem, abort) as state:
        state.check()
        for call in tool_calls:
            result = {"name": call.name, "call_id": call.id}
            try:
                invocation = _prepare(tools, blacklist, call, state)
                if invocation is None:
                    result["error"] = "Tool not registered or blacklisted"
                else:
                    admit(executor, invocation)
                    result["result"] = await executor.aexecute(invocation)
            except Exception as error:
                if not legacy:
                    raise
                result["error"] = str(error)
            state.record(call, result)
            state.check()
    return (
        [{k: v for k, v in item.items() if k != "call_id"} for item in mem.tool_results]
        if legacy
        else mem.tool_results
    )
