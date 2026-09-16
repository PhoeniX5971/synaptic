"""One-shot model tool dispatch through the executor boundary."""

from typing import Any, List
from .tool import Tool, ToolCall
from .invocation import LocalExecutor


def run_tools_sync(
    tools: List[Tool], blacklist: List[str], tool_calls: List[ToolCall], executor=None
) -> List[Any]:
    results = []
    tool_map = {t.name: t for t in tools}
    for call in tool_calls:
        if not isinstance(call, ToolCall):
            results.append({"error": "Invalid tool call format"})
            continue
        name, args = call.name, call.args
        tool = tool_map.get(name)
        if tool and name not in blacklist:
            if tool.is_async:
                raise RuntimeError(f"Tool '{name}' is async, cannot run in sync invoke()")
            try:
                results.append(
                    {
                        "name": name,
                        "result": (executor or LocalExecutor()).execute(tool.prepare(call)),
                    }
                )
            except Exception as e:
                results.append({"name": name, "error": str(e)})
        else:
            results.append({"name": name, "error": "Tool not registered or blacklisted"})
    return results


async def run_tools_async(
    tools: List[Tool], blacklist: List[str], tool_calls: List[ToolCall], executor=None
) -> List[Any]:
    results = []
    tool_map = {t.name: t for t in tools}
    for call in tool_calls:
        if not isinstance(call, ToolCall):
            results.append({"error": "Invalid tool call format"})
            continue
        name, args = call.name, call.args
        tool = tool_map.get(name)
        if tool and name not in blacklist:
            try:
                res = await (executor or LocalExecutor()).aexecute(tool.prepare(call))
                results.append({"name": name, "result": res})
            except Exception as e:
                results.append({"name": name, "error": str(e)})
        else:
            results.append({"name": name, "error": "Tool not registered or blacklisted"})
    return results
