from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .history import History
    from ..tool import Tool
    from ..capabilities import Capabilities
import enum

from .memory import ResponseMem
from .._metadata import finish_reason as normalize_finish


class ResponseFormat(enum.StrEnum):
    JSON = "json"
    NONE = "none"


@dataclass
class ToolCallArgsDelta:
    """Partial tool-call argument fragment streamed before the call completes."""

    name: str
    delta: str  # this token's fragment
    snapshot: str  # full args accumulated so far


@dataclass
class ResponseChunk:
    """Chunk emitted by async stream."""

    text: str
    is_final: bool = False
    function_call: Optional[Any] = None
    tool_call_delta: Optional[ToolCallArgsDelta] = None
    finish_reason: Optional[str] = None  # "stop" | "tool_calls" | "max_tokens"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    final: bool = False
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.extensions = dict(self.extensions)
        self.finish_reason = normalize_finish(self.finish_reason, self.extensions)


from ...providers._lifecycle import AdapterLifecycle


class BaseModel(AdapterLifecycle, ABC):
    history: "History | None"
    bound_tools: list["Tool"]
    synaptic_tools: list["Tool"]
    instructions: str
    temperature: float
    max_tokens: int
    response_format: ResponseFormat
    response_schema: Any
    capabilities: "Capabilities"

    def _invalidate_tools(self):
        raise NotImplementedError

    @abstractmethod
    def invoke(self, prompt: Optional[str], **kwargs) -> ResponseMem:
        pass

    async def astream(self, prompt: Optional[str], **kwargs):
        raise NotImplementedError("Provider does not support asynchronous streaming")
        yield
