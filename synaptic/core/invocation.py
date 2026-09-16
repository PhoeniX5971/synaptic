"""Immutable effective proposals and a public, injectable execution boundary."""

import copy
import inspect
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from ._awaitables import require_sync


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class Invocation:
    name: str
    call_id: str
    arguments: Any
    identity: Any = None
    _function: Any = field(default=None, repr=False)
    _values: Any = field(default=None, repr=False)
    _output: Any = field(default=None, repr=False)

    def execute(self):
        """Execute a captured callable with a private copy of the validated arguments."""
        return self._function(**copy.deepcopy(self._values))

    def validate_output(self, result):
        return self._output.validate_python(result) if self._output else result


class Executor(Protocol):
    def execute(self, invocation: Invocation) -> Any: ...
    async def aexecute(self, invocation: Invocation) -> Any: ...


class LocalExecutor:
    """Sequential inline executor, or explicit bounded thread scheduling."""

    def __init__(self, scheduler=None):
        self.scheduler = scheduler

    def execute(self, invocation):
        value = require_sync(invocation.execute(), "Async tool requires async execution")
        return invocation.validate_output(value)

    async def aexecute(self, invocation):
        if self.scheduler and not inspect.iscoroutinefunction(invocation._function):
            value = await self.scheduler.run(invocation.execute)
        else:
            value = invocation.execute()
        if inspect.isawaitable(value):
            value = await value
        return invocation.validate_output(value)
