from typing import List, Any
from ..providers import GeminiAdapter, OpenAIAdapter

from .base import BaseModel, History, ResponseMem
from .provider import Provider
from .tool import Tool


class Model:
    def __init__(
        self,
        provider: Provider,
        model: str,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        tools: List[Tool] = None,  # type: ignore
        history: History = None,  # type: ignore
        autorun: bool = False,
        automem: bool = False,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools or []
        self.history: History = history or History()
        self.autorun = autorun
        self.automem = automem
        self._initiate_model()

    def bind_tools(self, tools: List[Tool]):
        self.tools += tools

    def _initiate_model(self) -> BaseModel:
        if self.provider == Provider.OPENAI:
            return OpenAIAdapter(model=self.model, tools=self.tools)
        elif self.provider == Provider.GEMINI:
            return GeminiAdapter(
                self.model,
                temperature=self.temperature,
                tools=self.tools,
                history=self.history,
            )
        else:
            return GeminiAdapter(
                self.model,
                temperature=self.temperature,
                tools=self.tools,
                history=self.history,
            )

    def _run_tools(self, tool_calls: List[dict[str, Any]]) -> List[Any]:
        """Run tool calls returned by the model."""
        results = []
        tool_map = {tool.name: tool.function for tool in self.tools}

        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {})
            if name in tool_map:
                try:
                    result = tool_map[name](**args)
                    results.append({"name": name, "result": result})
                except Exception as e:
                    results.append({"name": name, "error": str(e)})
            else:
                results.append({"name": name, "error": "Tool not registered"})

        return results

    def invoke(
        self, prompt: str, autorun: bool = None, automem: bool = None, **kwargs  # type: ignore
    ) -> ResponseMem:
        llm = self._initiate_model()
        memory = llm.invoke(prompt, **kwargs)

        autorun = autorun if (autorun is not None) else self.autorun
        automem = automem if (automem is not None) else self.automem

        if autorun:
            if memory.tool_calls:
                tool_results = self._run_tools(memory.tool_calls)
                # Attach results for downstream use
                memory.tool_results = tool_results  # optional extra field
        else:
            memory.tool_results = []

        if automem:
            self.history.add(memory)

        return memory
