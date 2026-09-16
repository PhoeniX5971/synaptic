"""Lifecycle and external execution integration shared by Model instances."""

from .base.memory import complete_tool_call


class ModelAPI:
    def propose(self, prompt=None, **kwargs):
        """Generate correlated tool proposals without executing effects."""
        kwargs["autorun"] = False
        return self.invoke(prompt, **kwargs)

    async def apropose(self, prompt=None, **kwargs):
        kwargs["autorun"] = False
        return await self.ainvoke(prompt, **kwargs)

    def supply_results(self, response, results):
        """Supply external results keyed by native/stable call_id."""
        complete_tool_call(response, results, self.history)

    def close(self):
        self.llm.close()

    async def aclose(self):
        await self.llm.aclose()
