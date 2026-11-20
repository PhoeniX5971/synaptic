import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Literal
from uuid import uuid4

from ..core.base import History


@dataclass
class Session:
    """State container for an agent run or conversation."""

    id: str = field(default_factory=lambda: uuid4().hex)
    history: History = field(default_factory=History)
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Literal["idle", "running", "done", "failed", "cancelled", "exhausted"] = "idle"

    outcome: str | None = None
    # Canonical task facts, kept apart from the conversation. JSON-compatible values.
    task: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "id": self.id,
            "history": self.history.to_dict(),
            "metadata": copy.deepcopy(self.metadata),
            "task": copy.deepcopy(self.task),
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Session":
        """Restore an idle session; a session saved while running is not resumed as running."""
        if value.get("version") != 1:
            raise ValueError("Unsupported session serialization version")
        return cls(
            id=value["id"],
            history=History.from_dict(value["history"]),
            metadata=copy.deepcopy(value.get("metadata", {})),
            outcome=value.get("outcome"),
            task=copy.deepcopy(value.get("task", {})),
        )

    def reset(self) -> None:
        """Reset to idle so the session can be reused. Raises if running."""
        if self.state == "running":
            raise RuntimeError("Cannot reset a running session")
        self.state = "idle"
        self.outcome = None


class BudgetExceeded(RuntimeError):
    """The turn or protocol retry budget was exhausted."""
