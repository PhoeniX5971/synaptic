import asyncio
import time
from datetime import datetime, timezone
from typing import Any, List, Optional

from .base import ResponseMem, UserMem
from ._awaitables import require_sync
from ._completion import Completion
from .tool import Tool, ToolCall
from .invocation import LocalExecutor
from .base.content import media_content
from .capabilities import Capabilities, UnsupportedCapability
from ..logging import _bus


from ._tool_execution import run_tools_sync, run_tools_async


def _ctx(model) -> list:
    return [{"role": m.role, "content": m.message} for m in model.history.entries]


def _check(model, mode, images, audio):
    capabilities = getattr(model.llm, "capabilities", None)
    if isinstance(capabilities, Capabilities):
        capabilities.check(
            mode,
            images=images,
            audio=audio,
            structured=getattr(model, "response_format", "none") != "none",
        )
        if model.llm.synaptic_tools and not capabilities.tools:
            raise UnsupportedCapability("This adapter does not implement tools")
        for memory in model.history.entries:
            for block in getattr(memory, "content", ()):
                if block.kind == "resource" or (
                    block.kind in ("image", "audio")
                    and not getattr(
                        capabilities, block.kind + "s" if block.kind == "image" else "audio"
                    )
                ):
                    raise UnsupportedCapability("History contains unsupported media")
            if getattr(memory, "tool_calls", None) and len(memory.tool_results) != len(
                memory.tool_calls
            ):
                raise ValueError("Supply every tool result before continuing this history")


def invoke(
    model,
    prompt: Optional[str],
    role: str = "user",
    images=None,
    audio=None,
    autorun: bool | None = None,
    automem: bool | None = None,
    **kwargs,
) -> ResponseMem:
    _check(model, "invoke", images, audio)
    if role not in ("user", "assistant", "system"):
        raise ValueError("Role must be one of 'user', 'assistant', or 'system'")

    tools = list(getattr(model.llm, "synaptic_tools", ()))
    created = datetime.now().astimezone(timezone.utc)
    t0 = time.monotonic()
    memory = model.llm.invoke(prompt=prompt, role=role, images=images, audio=audio, **kwargs)

    if _bus.active():
        _bus.emit(
            {
                "type": "model.invoke",
                "provider": str(model.provider),
                "model": model.model,
                "prompt": prompt,
                "role": role,
                "context": _ctx(model),
                "response": memory.message,
                "input_tokens": memory.input_tokens,
                "output_tokens": memory.output_tokens,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }
        )

    _autorun = autorun if autorun is not None else model.autorun
    _automem = automem if automem is not None else model.automem

    if _autorun and memory.tool_calls:
        if any(t.is_async for t in model.llm.synaptic_tools):
            raise RuntimeError("invoke() cannot run async tools; use ainvoke()")
        memory.tool_results = run_tools_sync(
            tools,
            model.blacklist,
            memory.tool_calls,
            getattr(model, "executor", None),
        )
    else:
        memory.tool_results = []

    if _automem and model.history is not None:
        if prompt is not None or images or audio:
            model.history.add(
                UserMem(
                    message=prompt or "",
                    role=role,
                    created=created,
                    content=media_content(images, audio),
                )
            )
        model.history.add(memory)

    return memory


async def ainvoke(
    model,
    prompt: Optional[str],
    role: str = "user",
    images=None,
    audio=None,
    autorun: bool | None = None,
    automem: bool | None = None,
    **kwargs,
) -> ResponseMem:
    _check(model, "ainvoke", images, audio)
    if role not in ("user", "assistant", "system"):
        raise ValueError("Role must be one of 'user', 'assistant', or 'system'")

    tools = list(getattr(model.llm, "synaptic_tools", ()))
    created = datetime.now().astimezone(timezone.utc)
    _autorun = autorun if autorun is not None else model.autorun
    _automem = automem if automem is not None else model.automem

    t0 = time.monotonic()
    completion = Completion()
    async for chunk in model.llm.astream(
        prompt=prompt, role=role, images=images, audio=audio, **kwargs
    ):
        completion.add(chunk)

    memory = completion.memory(created)

    if _bus.active():
        _bus.emit(
            {
                "type": "model.ainvoke",
                "provider": str(model.provider),
                "model": model.model,
                "prompt": prompt,
                "role": role,
                "context": _ctx(model),
                "response": completion.text,
                "input_tokens": memory.input_tokens,
                "output_tokens": memory.output_tokens,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }
        )

    if _autorun and memory.tool_calls:
        memory.tool_results = await run_tools_async(
            tools,
            model.blacklist,
            memory.tool_calls,
            getattr(model, "executor", None),
        )
    else:
        memory.tool_results = []

    if _automem and model.history is not None:
        if prompt is not None or images or audio:
            model.history.add(
                UserMem(
                    message=prompt or "",
                    role=role,
                    created=created,
                    content=media_content(images, audio),
                )
            )
        model.history.add(memory)

    return memory


from ._model_stream import astream
