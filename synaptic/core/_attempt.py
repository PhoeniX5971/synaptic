"""Retry owns only invisible, effect-free generation attempts, never prior history."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from .base import ResponseMem
from .base.content import media_content

_current: ContextVar["Attempt | None"] = ContextVar("synaptic_generation_attempt", default=None)


class Attempt:
    def __init__(self, model):
        self.model = model
        self.started = self.output = self.recorded = False
        self.inputs = {}

    def archive(self, memory):
        self.recorded = True
        memory.extensions["attempt_input"] = self.inputs
        self.model.history.failed_attempts.append(memory)
        self.model.last_response = memory


@contextmanager
def attempt(model):
    # Generic retry targets retain their existing behavior; only Model opts into ownership.
    from .model import Model

    if not isinstance(model, Model):
        yield None
        return
    state = Attempt(model)
    token = _current.set(state)
    try:
        yield state
    except BaseException as error:
        if state.started and not state.recorded and not state.output:
            memory = ResponseMem(
                "",
                datetime.now(timezone.utc),
                extensions={"incomplete": True, "execution_error": type(error).__name__},
            )
            state.archive(memory)
        raise
    finally:
        _current.reset(token)


def begin(model, prompt, role, images, audio):
    state = _current.get()
    if state is not None and state.model is model:
        state.started = True
        state.inputs = {
            "prompt": prompt,
            "role": role,
            "content": [block.to_dict() for block in media_content(images, audio)],
        }


def observe(model):
    state = _current.get()
    if state is not None and state.model is model:
        state.output = True


def archive_failure(model, memory):
    state = _current.get()
    if state is None or state.model is not model:
        return False
    state.recorded = True
    if (
        memory.extensions.get("incomplete")
        and not state.output
        and not memory.tool_calls
        and not memory.tool_results
    ):
        state.archive(memory)
        return True
    return False
