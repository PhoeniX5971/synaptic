from typing import Any, List, Optional

from . import _runner as _r
from ._factory import build_adapter
from .base import History, ResponseFormat, ResponseMem
from .provider import Provider
from .tool import Tool, ToolRegistry


class Model:
    """Provider-neutral interface for one LLM configuration.

    `Model` owns provider selection, prompt invocation, streaming delegation,
    optional memory writes, optional one-shot tool execution, and dynamic
    `history`/`instructions` updates. Use `Agent` when you want a multi-turn
    loop that continues after tools return results.
    """

    def __init__(
        self,
        provider: Provider,
        model: str,
        temperature: float = 0.8,
        api_key: str = "",
        max_tokens: int = 1024,
        tools: Optional[List[Tool]] = None,
        history: Optional[History] = None,
        autorun: bool = False,
        automem: bool = False,
        blacklist: List[str] | None = None,
        location: Optional[str] = None,
        project: Optional[str] = None,
        instructions: str = "",
        response_format: ResponseFormat = ResponseFormat.NONE,
        response_schema: Any = None,
        base_url: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.autorun = autorun
        self.automem = automem
        self.blacklist = blacklist or []
        self.response_format = response_format
        self.response_schema = response_schema
        if self.response_format != ResponseFormat.NONE and self.response_schema is None:
            raise ValueError("Response schema must be provided for structured response formats")
        self.location = location
        self.project = project
        self.base_url = base_url
        self.tool_registry = tool_registry
        self._instructions = instructions
        self._history = history or History()

        adapter_tools = (tools or []) if self.response_format == ResponseFormat.NONE else None
        self.llm = build_adapter(
            provider=provider,
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
            tools=adapter_tools,
            history=self._history,
            response_format=response_format,
            response_schema=response_schema,
            instructions=instructions,
            location=location,
            project=project,
            base_url=base_url,
            tool_registry=tool_registry,
        )
        self.llm._invalidate_tools()
        self.tools = self.llm.synaptic_tools

        if not self.automem:
            self._history.window(1)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def history(self) -> History:
        return self._history

    @history.setter
    def history(self, value: History) -> None:
        self._history = value
        if hasattr(self, "llm"):
            self.llm.history = value

    @property
    def instructions(self) -> str:
        return self._instructions

    @instructions.setter
    def instructions(self, value: str) -> None:
        self._instructions = value
        if hasattr(self, "llm"):
            self.llm.instructions = value

    # ------------------------------------------------------------------ #
    # Tool binding                                                         #
    # ------------------------------------------------------------------ #

    def bind_tools(self, tools: List[Tool]) -> None:
        """Attach tools directly to this model and refresh adapter schemas."""
        if self.tools is None:
            self.tools = []
        self.tools += tools
        self.llm.synaptic_tools = self.tools
        self.llm._invalidate_tools()

    # ------------------------------------------------------------------ #
    # Invoke / stream delegation                                           #
    # ------------------------------------------------------------------ #

    def invoke(
        self, prompt: Optional[str], role: str = "user",
        images: Optional[List[str]] = None, audio: Optional[List[str]] = None,
        autorun: bool = None, automem: bool = None, **kwargs,
    ) -> ResponseMem:
        """Run one synchronous provider call and return a `ResponseMem`."""
        return _r.invoke(self, prompt, role=role, images=images, audio=audio,
                         autorun=autorun, automem=automem, **kwargs)

    async def ainvoke(
        self, prompt: Optional[str], role: str = "user",
        images: Optional[List[str]] = None, audio: Optional[List[str]] = None,
        autorun: bool = None, automem: bool = None, **kwargs,
    ) -> ResponseMem:
        """Run one provider call from async code and return a `ResponseMem`."""
        return await _r.ainvoke(self, prompt, role=role, images=images, audio=audio,
                                autorun=autorun, automem=automem, **kwargs)

    async def astream(
        self, prompt: Optional[str], role: str = "user",
        images: Optional[List[str]] = None, audio: Optional[List[str]] = None,
        autorun: bool = None, automem: bool = None, abort=None, **kwargs,
    ):
        """Yield provider stream chunks, optionally stopping on `abort`."""
        async for chunk in _r.astream(self, prompt, role=role, images=images, audio=audio,
                                      autorun=autorun, automem=automem, abort=abort, **kwargs):
            yield chunk
