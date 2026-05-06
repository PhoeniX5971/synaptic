import inspect
from datetime import datetime, timezone
from typing import Any, List, Optional

from .base import ResponseMem, UserMem
from .tool import Tool, ToolCall


def run_tools_sync(tools: List[Tool], blacklist: List[str], tool_calls: List[ToolCall]) -> List[Any]:
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
                results.append({"name": name, "result": tool.run(**args)})
            except Exception as e:
                results.append({"name": name, "error": str(e)})
        else:
            results.append({"name": name, "error": "Tool not registered or blacklisted"})
    return results


async def run_tools_async(tools: List[Tool], blacklist: List[str], tool_calls: List[ToolCall]) -> List[Any]:
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
                res = tool.run(**args)
                if inspect.iscoroutine(res):
                    res = await res
                results.append({"name": name, "result": res})
            except Exception as e:
                results.append({"name": name, "error": str(e)})
        else:
            results.append({"name": name, "error": "Tool not registered or blacklisted"})
    return results


def invoke(model, prompt: Optional[str], role: str = "user", images=None, audio=None,
           autorun: bool = None, automem: bool = None, **kwargs) -> ResponseMem:
    if role not in ("user", "assistant", "system"):
        raise ValueError("Role must be one of 'user', 'assistant', or 'system'")

    created = datetime.now().astimezone(timezone.utc)
    memory = model.llm.invoke(prompt=prompt, role=role, images=images, audio=audio, **kwargs)

    _autorun = autorun if autorun is not None else model.autorun
    _automem = automem if automem is not None else model.automem

    if _autorun and memory.tool_calls:
        if any(t.is_async for t in model.llm.synaptic_tools):
            raise RuntimeError("invoke() cannot run async tools; use ainvoke()")
        memory.tool_results = run_tools_sync(model.llm.synaptic_tools, model.blacklist, memory.tool_calls)
    else:
        memory.tool_results = []

    if _automem and model.history and prompt is not None:
        model.history.add(UserMem(message=prompt, role=role, created=created))
        model.history.add(memory)

    return memory


async def ainvoke(model, prompt: Optional[str], role: str = "user", images=None, audio=None,
                  autorun: bool = None, automem: bool = None, **kwargs) -> ResponseMem:
    if role not in ("user", "assistant", "system"):
        raise ValueError("Role must be one of 'user', 'assistant', or 'system'")

    created = datetime.now().astimezone(timezone.utc)
    _autorun = autorun if autorun is not None else model.autorun
    _automem = automem if automem is not None else model.automem

    # Drive the native async stream to completion and collect the full response.
    # This gives true async execution without blocking the event loop.
    accumulated = ""
    tool_calls: List[ToolCall] = []
    async for chunk in model.llm.astream(prompt=prompt, role=role, images=images, audio=audio, **kwargs):
        if not chunk.is_final and chunk.text:
            accumulated += chunk.text
        if chunk.function_call:
            tool_calls.append(chunk.function_call)

    memory = ResponseMem(message=accumulated, created=created, tool_calls=tool_calls)

    if _autorun and memory.tool_calls:
        memory.tool_results = await run_tools_async(model.llm.synaptic_tools, model.blacklist, memory.tool_calls)
    else:
        memory.tool_results = []

    if _automem and model.history and prompt is not None:
        model.history.add(UserMem(message=prompt, role=role, created=created))
        model.history.add(memory)

    return memory


async def astream(model, prompt: Optional[str], role: str = "user", images=None, audio=None,
                  autorun: bool = None, automem: bool = None, abort=None, **kwargs):
    if role not in ("user", "assistant", "system"):
        raise ValueError("Role must be one of 'user', 'assistant', or 'system'")

    if not hasattr(model.llm, "astream"):
        raise NotImplementedError("Underlying model does not implement astream()")

    _autorun = autorun if autorun is not None else model.autorun
    _automem = automem if automem is not None else model.automem

    created = datetime.now().astimezone(timezone.utc)
    accumulated = ""
    tool_calls: List[ToolCall] = []
    tool_results: List[Any] = []

    async for chunk in model.llm.astream(prompt=prompt, role=role, images=images, audio=audio, abort=abort, **kwargs):
        if abort and abort.is_set():
            return
        yield chunk

        if getattr(chunk, "text", None):
            accumulated += chunk.text

        if getattr(chunk, "function_call", None):
            tool_calls.append(chunk.function_call)
            if _autorun:
                try:
                    tr = await run_tools_async(model.llm.synaptic_tools, model.blacklist, [chunk.function_call])
                    tool_results.extend(tr)
                except Exception as e:
                    tool_results.append({"name": chunk.function_call.name, "error": str(e)})

    final_mem = ResponseMem(message=accumulated, created=created, tool_calls=tool_calls)
    final_mem.tool_results = tool_results

    if _automem and model.history and prompt is not None:
        model.history.add(UserMem(message=prompt, role=role, created=created))
        model.history.add(final_mem)
