"""Lifecycle and external execution integration shared by Model instances."""

from .base.memory import complete_tool_call
from .base._results import ResultInput


from .base import BaseModel, History, ResponseMem


class ModelAPI:
    llm: BaseModel

    history: History
    last_response: ResponseMem | None = None

    def invoke(self, prompt: str | None, **kwargs) -> ResponseMem:
        raise NotImplementedError

    async def ainvoke(self, prompt: str | None, **kwargs) -> ResponseMem:
        raise NotImplementedError

    def propose(self, prompt: str | None = None, **kwargs) -> ResponseMem:
        """Generate correlated tool proposals without executing effects."""
        kwargs["autorun"] = False
        return self.invoke(prompt, **kwargs)

    async def apropose(self, prompt: str | None = None, **kwargs) -> ResponseMem:
        kwargs["autorun"] = False
        return await self.ainvoke(prompt, **kwargs)

    def supply_results(self, response: ResponseMem, results: ResultInput) -> None:
        """Supply external results keyed by native/stable call_id."""
        complete_tool_call(response, results, self.history)

    def close(self):
        self.llm.close()

    async def aclose(self):
        await self.llm.aclose()
