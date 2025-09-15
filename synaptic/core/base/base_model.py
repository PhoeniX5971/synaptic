from abc import ABC, abstractmethod

from .memory import ResponseMem


class BaseModel(ABC):
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> ResponseMem:
        pass
