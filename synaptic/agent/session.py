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

    def reset(self) -> None:
        """Reset to idle so the session can be reused. Raises if running."""
        if self.state == "running":
            raise RuntimeError("Cannot reset a running session")
        self.state = "idle"
        self.outcome = None


class BudgetExceeded(RuntimeError):
    """The turn or protocol retry budget was exhausted."""
