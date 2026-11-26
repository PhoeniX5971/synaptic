"""Skills: named bundles of instructions and tools the agent loads when a task needs them."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence

from ..core.tool import Tool
from ._extend import Extension
from .final import FinalAnswer
from .todos import Todo, Todos


@dataclass
class Skill:
    name: str
    description: str  # Always visible to the model, in the catalog.
    instructions: str = ""  # Visible once the skill is loaded.
    tools: Sequence[Tool] = ()
    todos: Sequence[Todo] = ()  # Added to the todo list when the skill loads.
    final: Any = None  # Schema of this skill's own run-ending result tool, final_<name>.
    final_required: bool = False


class Skills(Extension):
    """Offer skills through a `load_skill` tool. Any number can be loaded and they compose.

    Loaded skills stay loaded for the life of the session. With `preload_tools=True`
    every skill's tools are declared from the first turn, so the tool list never
    changes (better for provider prompt caching, more input tokens); only the
    instructions load on demand.
    """

    name = "skills"

    def __init__(self, skills: Iterable[Skill], preload_tools: bool = False):
        self._skills = {skill.name: skill for skill in skills}
        self.preload_tools = preload_tools
        self._finals = {
            name: FinalAnswer(
                skill.final,
                name=f"final_{name}",
                description=f"Give the final result of the {name} skill and end the task. Call it alone.",
                required=skill.final_required,
                allowed=self._gate(name),
            )
            for name, skill in self._skills.items()
            if skill.final is not None
        }
        parameters = {
            "type": "object",
            "properties": {"name": {"type": "string", "enum": list(self._skills)}},
            "required": ["name"],
        }
        description = "Load a skill: its instructions and tools become available from your next step."
        declaration = {"name": "load_skill", "description": description, "parameters": parameters}
        self._loader = Tool(
            "load_skill", declaration, self.load, add_to_registry=False, metadata={"errors": "return"}
        )

    def _gate(self, name: str) -> Callable[[], Optional[str]]:
        return lambda: None if name in self.loaded else f"Load the {name} skill first."

    def attach(self, agent) -> None:
        super().attach(agent)
        for final in self._finals.values():
            final.attach(agent)
        missing = [name for name in self.loaded if name not in self._skills]
        if missing:
            raise ValueError(f"Session has loaded skills that were not provided: {missing}")
        needs_todos = [skill.name for skill in self._skills.values() if skill.todos]
        if needs_todos and self._todos() is None:
            raise ValueError(f"Skills {needs_todos} have todos; add Todos() to the extensions")
        if self.preload_tools:
            for name in self._skills:
                self._check_clash(name, self._names(set(self._skills) - {name}) | set(agent._registry.all()))

    @property
    def loaded(self) -> List[str]:
        return list(self.state.get("loaded", []))

    def _todos(self) -> Optional[Todos]:
        return next((ext for ext in self.agent.extensions if isinstance(ext, Todos)), None)

    def _bundle(self, name: str) -> List[Tool]:
        final = self._finals.get(name)
        return [*self._skills[name].tools, *(final.tools() if final else [])]

    def _names(self, skills: Iterable[str]) -> set:
        return {tool.name for name in skills for tool in self._bundle(name)}

    def _check_clash(self, name: str, taken: set) -> None:
        clash = sorted({tool.name for tool in self._bundle(name)} & taken)
        if clash:
            raise ValueError(f"Skill {name} cannot load: tool names already in use: {clash}")

    def load(self, name: str) -> str:
        """Load a skill by name (also the `load_skill` tool). Loading twice is a no-op."""
        if name not in self._skills:
            raise ValueError(f"Unknown skill {name!r}. Available: {', '.join(self._skills)}.")
        if name in self.loaded:
            return f"Skill {name} is already loaded."
        if not self.preload_tools:
            self._check_clash(name, self.agent.tool_names())
        self.state.setdefault("loaded", []).append(name)
        todos = self._todos()
        for index, todo in enumerate(self._skills[name].todos, 1):
            todos.add(todo, id=todo.id or f"{name}:{index}")  # type: ignore[union-attr]
        self.agent.notify("skill_loaded", name)
        return f"Skill {name} loaded. Its instructions and tools are available from your next step."

    def tools(self) -> List[Tool]:
        active = self._skills if self.preload_tools else self.loaded
        return [self._loader, *(tool for name in active for tool in self._bundle(name))]

    def instructions(self) -> str:
        catalog = [f"- {skill.name}: {skill.description}" for skill in self._skills.values()]
        parts = [
            "Skills you can load with the load_skill tool when a task needs them "
            "(several can be loaded together):\n" + "\n".join(catalog)
        ]
        for name in self.loaded:  # Appended in load order; earlier text never changes.
            if self._skills[name].instructions:
                parts.append(f"## Skill: {name}\n{self._skills[name].instructions}")
        return "\n\n".join(parts)

    def before_finish(self, final: bool = False) -> Optional[str]:
        required = [f"final_{name}" for name in self.loaded if self._skills[name].final_required]
        if final or not required:
            return None
        return "Finish by calling " + " or ".join(required) + "."
