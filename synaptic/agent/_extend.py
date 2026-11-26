"""Extension seam: opt-in agent behaviour built only on the agent's public API."""

from typing import Any, List, Optional

from ..core.base import Memory
from ..core.tool import Tool
from ._state import AgentState


class Extension:
    """Base class for agent extensions. Override only what you need.

    One instance belongs to one agent. `state` is this extension's JSON-compatible
    dict on the session, saved and restored with it under `name`.
    """

    name: str = ""
    agent: Any = None

    def attach(self, agent) -> None:
        if self.agent is not None and self.agent is not agent:
            raise RuntimeError("An extension instance belongs to one agent")
        self.agent = agent

    def tools(self) -> List[Tool]:
        """Tools offered to the model; read again at every turn."""
        return []

    def instructions(self) -> str:
        """Text appended to the agent's instructions; read again at every turn."""
        return ""

    def context(self, entries: List[Memory]) -> List[Memory]:
        """Per-request view of the context; never stored in history."""
        return entries

    def before_finish(self, final: bool = False) -> Optional[str]:
        """Return a message to refuse finishing; `final` is True for a final-tool call."""
        return None

    @property
    def state(self) -> dict:
        return self.agent.session.extensions.setdefault(self.name, {})


class AgentExtensible(AgentState):
    extensions: tuple = ()
    pending_calls: tuple = ()  # Calls of the response currently being dispatched.
    _owns_instructions = False
    _final: Any = None
    _outbox: list

    def add_tools(self, *tools: Tool) -> None:
        """Offer more tools from the next turn on."""
        for tool in tools:
            self._registry.register(tool)

    def remove_tools(self, *names: str) -> None:
        for name in names:
            self._registry.unregister(name)

    def tool_names(self) -> set:
        return set(self._registry.all()) | set(self._extension_tools())

    @property
    def instructions(self) -> str:
        """The agent's base instructions; extensions append to them every turn."""
        return self._base_instructions

    @instructions.setter
    def instructions(self, value: str) -> None:
        self._base_instructions, self._owns_instructions = value or "", True

    def finish(self, output: Any, name: str = "final", data: Any = None) -> None:
        """Called from a tool: end the run after this tool batch with `output` as its result.

        `data` is the JSON-compatible form kept on the session; it defaults to `output`.
        """
        self._final = (name, output)
        self.session.extensions["final"] = {
            "name": name,
            "output": output if data is None else data,
        }

    def notify(self, event: str, *args: Any) -> None:
        """Queue an event from inside a tool; delivered on `events` after the tool batch."""
        self._outbox.append((event, args))

    def blockers(self, final: bool = False) -> List[str]:
        """Messages from extensions that refuse to let the run finish now."""
        found = (extension.before_finish(final) for extension in self.extensions)
        return [message for message in found if message]

    def _extension_tools(self) -> dict:
        return {tool.name: tool for ext in self.extensions for tool in ext.tools()}

    def _compose(self) -> str:
        parts = [self._base_instructions, *(ext.instructions() for ext in self.extensions)]
        return "\n\n".join(part for part in parts if part)

    def _project(self, entries: List[Memory]) -> List[Memory]:
        for extension in self.extensions:
            entries = extension.context(entries)
        return entries

    def _conclude(self, memory) -> bool:
        """Attach a tool-requested result to the response that ends the run."""
        if self._final is None:
            return False
        memory.output_name, memory.output = self._final
        return True
