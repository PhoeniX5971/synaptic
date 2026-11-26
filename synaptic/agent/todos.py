"""A plan the agent keeps, with required items that must be done before it may finish."""

from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional

from ..core.base import Memory, UserMem
from ..core.tool import Tool
from ._extend import Extension

STATUSES = ("open", "in_progress", "done", "skipped")
_MARK = {"open": "[ ]", "in_progress": "[~]", "done": "[x]", "skipped": "[-]"}


@dataclass
class Todo:
    text: str
    required: bool = False
    status: str = "open"
    note: str = ""
    skippable: bool = False
    id: str = ""


class Todos(Extension):
    """Todo list with `todo_add`, `todo_update` and `todo_list` tools.

    `placement="message"` (default) shows the list as the last message of each request,
    leaving the system prompt stable for provider prompt caching. `"instructions"`
    appends it to the system prompt instead. Neither is stored in history.
    """

    name = "todos"

    def __init__(self, items: Iterable[Todo] = (), placement: str = "message"):
        if placement not in ("message", "instructions"):
            raise ValueError("placement must be message or instructions")
        self._initial, self.placement = list(items), placement
        self._tools = [
            self._tool("todo_add", "Add an item to your todo list.", self._add, {"text": {"type": "string"}}, ["text"]),
            self._tool(
                "todo_update",
                "Change the status of a todo item, optionally with a short note.",
                self._update,
                {
                    "id": {"type": "string"},
                    "status": {"type": "string", "enum": list(STATUSES)},
                    "note": {"type": "string"},
                },
                ["id", "status"],
            ),
            self._tool("todo_list", "Show the todo list.", self.render, {}, []),
        ]

    @staticmethod
    def _tool(name, description, function, properties, required):
        parameters = {"type": "object", "properties": properties, "required": required}
        declaration = {"name": name, "description": description, "parameters": parameters}
        return Tool(name, declaration, function, add_to_registry=False, metadata={"errors": "return"})

    def attach(self, agent) -> None:
        super().attach(agent)
        for index, todo in enumerate(self._initial, 1):
            self.add(todo, id=todo.id or f"r{index}", announce=False)

    @property
    def items(self) -> List[Todo]:
        return [Todo(**item) for item in self.state.get("items", [])]

    def open_required(self) -> List[Todo]:
        return [item for item in self.items if item.required and item.status not in ("done", "skipped")]

    def add(self, todo: Todo, id: Optional[str] = None, announce: bool = True) -> Todo:
        """Add an item; an id already on the session is kept as it is (restored sessions)."""
        stored = self.state.setdefault("items", [])
        if id is None:
            self.state["next"] = number = self.state.get("next", 0) + 1
            id = f"t{number}"
        existing = next((item for item in stored if item["id"] == id), None)
        if existing is not None:
            return Todo(**existing)
        if todo.status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        item = {**asdict(todo), "id": id}
        stored.append(item)
        if announce:
            self.agent.notify("todo_added", dict(item))
        return Todo(**item)

    def render(self) -> str:
        lines = ["Todo list (keep it current with todo_update):"]
        for item in self.items:
            flag = " (required)" if item.required else ""
            note = f" - {item.note}" if item.note else ""
            lines.append(f"{_MARK[item.status]} {item.id}{flag}: {item.text}{note}")
        return "\n".join(lines) if len(lines) > 1 else "The todo list is empty."

    def tools(self) -> List[Tool]:
        return self._tools

    def instructions(self) -> str:
        return self.render() if self.placement == "instructions" and self.items else ""

    def context(self, entries: List[Memory]) -> List[Memory]:
        if self.placement != "message" or not self.items:
            return entries
        return [*entries, UserMem(self.render(), None)]

    def before_finish(self, final: bool = False) -> Optional[str]:
        missing = self.open_required()
        if not missing:
            return None
        listed = "; ".join(f"{item.id}: {item.text}" for item in missing)
        return f"Required todo items are not done: {listed}. Do them and mark them done first."

    def _add(self, text: str) -> str:
        return f"Added as {self.add(Todo(text)).id}."

    def _update(self, id: str, status: str, note: str = "") -> str:
        item = next((item for item in self.state.get("items", []) if item["id"] == id), None)
        if item is None:
            raise ValueError(f"No todo item with id {id!r}.")
        if status == "skipped" and item["required"] and not item["skippable"]:
            raise ValueError(f"{id} is required and cannot be skipped.")
        item.update(status=status, note=note or item["note"])
        self.agent.notify("todo_updated", dict(item))
        return f"{id} is now {status}."
