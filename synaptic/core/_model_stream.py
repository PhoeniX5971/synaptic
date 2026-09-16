"""Stream bookkeeping, partial history and deferred tool effects."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional
from ._completion import Completion
from ..logging import _bus
from ._runner import _check, _ctx, run_tools_async, remember
from ._execution_state import ExecutionState, check_cancelled


async def astream(
    model,
    prompt: Optional[str],
    role: str = "user",
    images=None,
    audio=None,
    autorun: bool | None = None,
    automem: bool | None = None,
    abort=None,
    **kwargs,
):
    check_cancelled(abort)
    _check(model, "astream", images, audio)
    if role not in ("user", "assistant", "system"):
        raise ValueError("Role must be one of 'user', 'assistant', or 'system'")

    if not hasattr(model.llm, "astream"):
        raise NotImplementedError("Underlying model does not implement astream()")

    _autorun = autorun if autorun is not None else model.autorun
    _automem = automem if automem is not None else model.automem

    if _bus.active():
        _bus.emit(
            {
                "type": "model.stream.start",
                "provider": str(model.provider),
                "model": model.model,
                "prompt": prompt,
                "role": role,
                "context": _ctx(model),
            }
        )

    tools = list(getattr(model.llm, "synaptic_tools", ()))
    created = datetime.now().astimezone(timezone.utc)
    t0 = time.monotonic()
    completion = Completion()

    source = model.llm.astream(
        prompt=prompt, role=role, images=images, audio=audio, abort=abort, **kwargs
    )
    try:
        async for chunk in source:
            if abort and abort.is_set():
                raise asyncio.CancelledError()
            completion.add(chunk)
            if not chunk.is_final and chunk.text:
                if _bus.stream_chunks and _bus.active():
                    _bus.emit({"type": "model.stream.chunk", "text": chunk.text})
            yield chunk
    except BaseException as error:
        partial = completion.memory(created)
        remember(model, partial, _automem, prompt, role, created, images, audio)
        ExecutionState(partial, abort).interrupt(error)
        raise
    finally:
        close = getattr(source, "aclose", None)
        if close:
            await close()

    if _bus.active():
        _bus.emit(
            {
                "type": "model.stream.end",
                "provider": str(model.provider),
                "model": model.model,
                "response": completion.text,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }
        )

    final_mem = completion.memory(created)
    remember(model, final_mem, _automem, prompt, role, created, images, audio)
    with ExecutionState(final_mem, abort):
        check_cancelled(abort)
        if _autorun and final_mem.tool_calls:
            await run_tools_async(
                tools,
                model.blacklist,
                final_mem.tool_calls,
                getattr(model, "executor", None),
                final_mem,
                abort,
            )
