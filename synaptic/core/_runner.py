import time
from datetime import datetime, timezone
from typing import Optional

from .base import ResponseMem, UserMem
from ._completion import Completion
from ._execution_state import ExecutionState, check_cancelled
from .base.content import media_content
from .capabilities import Capabilities, UnsupportedCapability
from ..logging import _bus


from ._tool_execution import run_tools_sync, run_tools_async


def _ctx(model) -> list:
    return [{"role": m.role, "content": m.message} for m in model.history.entries]


def _check(model, mode, images, audio):
    for memory in model.history.entries:
        if getattr(memory, "extensions", {}).get("incomplete") or (
            getattr(memory, "tool_calls", None)
            and len(memory.tool_results) != len(memory.tool_calls)
        ):
            raise ValueError(
                "Reconcile incomplete execution and supply every tool result before continuing"
            )
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


def invoke(
    model,
    prompt: Optional[str],
    role: str = "user",
    images=None,
    audio=None,
    autorun: bool | None = None,
    automem: bool | None = None,
    abort=None,
    **kwargs,
) -> ResponseMem:
    check_cancelled(abort)
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

    remember(model, memory, _automem, prompt, role, created, images, audio)
    with ExecutionState(memory, abort):
        check_cancelled(abort)
        if _autorun and memory.tool_calls:
            run_tools_sync(
                tools,
                model.blacklist,
                memory.tool_calls,
                getattr(model, "executor", None),
                memory,
                abort,
            )

    return memory


async def ainvoke(
    model,
    prompt: Optional[str],
    role: str = "user",
    images=None,
    audio=None,
    autorun: bool | None = None,
    automem: bool | None = None,
    abort=None,
    **kwargs,
) -> ResponseMem:
    check_cancelled(abort)
    _check(model, "ainvoke", images, audio)
    if role not in ("user", "assistant", "system"):
        raise ValueError("Role must be one of 'user', 'assistant', or 'system'")

    tools = list(getattr(model.llm, "synaptic_tools", ()))
    created = datetime.now().astimezone(timezone.utc)
    _autorun = autorun if autorun is not None else model.autorun
    _automem = automem if automem is not None else model.automem

    t0 = time.monotonic()
    completion = Completion()
    source = model.llm.astream(prompt=prompt, role=role, images=images, audio=audio, **kwargs)
    try:
        async for chunk in source:
            completion.add(chunk)
            check_cancelled(abort)
    except BaseException as error:
        memory = completion.memory(created)
        remember(model, memory, _automem, prompt, role, created, images, audio)
        ExecutionState(memory, abort).interrupt(error)
        raise
    finally:
        close = getattr(source, "aclose", None)
        if close:
            await close()
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

    remember(model, memory, _automem, prompt, role, created, images, audio)
    with ExecutionState(memory, abort):
        check_cancelled(abort)
        if _autorun and memory.tool_calls:
            await run_tools_async(
                tools,
                model.blacklist,
                memory.tool_calls,
                getattr(model, "executor", None),
                memory,
                abort,
            )

    return memory


def remember(model, memory, automem, prompt, role, created, images, audio):
    model.last_response = memory
    if automem and model.history is not None:
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


from ._model_stream import astream
