"""Expose an agent as a tool so another agent can delegate a task to it."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from ..core.tool import Tool

_abort: ContextVar[Any] = ContextVar("synaptic_agent_abort", default=None)


@contextmanager
def abort_scope(abort):
    """Expose the running agent's abort event to delegation tools it dispatches."""
    token = _abort.set(abort)
    try:
        yield
    finally:
        _abort.reset(token)


def subagent(name: str, description: str, source: Any, *, sync: bool = False) -> Tool:
    """Build a delegation tool from an Agent or a zero-argument Agent factory.

    A factory yields a fresh agent per call and is marked parallel-safe. A single
    Agent is reused: it keeps its history between delegations and runs one at a
    time. The delegate needs its own Model instance, session and budgets.
    """
    factory = not hasattr(source, "arun")

    def pick():
        agent: Any = source() if factory else source
        if agent.session.state != "idle":
            agent.session.reset()
        return agent

    async def delegate(task: str) -> str:
        return (await pick().arun(task, abort=_abort.get())).message

    def delegate_sync(task: str) -> str:
        return pick().run(task, abort=_abort.get()).message

    declaration = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Complete, self-contained task."}
            },
            "required": ["task"],
        },
    }
    return Tool(
        name,
        declaration,
        delegate_sync if sync else delegate,
        add_to_registry=False,
        metadata={"subagent": True, "parallel": factory},
    )
