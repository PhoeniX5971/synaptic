from abc import ABC, abstractmethod

from .memory import ResponseMem
import enum


class ResponseFormat(enum.StrEnum):
    JSON = "json"
    NONE = "none"


class BaseModel(ABC):
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> ResponseMem:
        pass
