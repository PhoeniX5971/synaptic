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
    state: Literal["idle", "running", "done"] = "idle"
