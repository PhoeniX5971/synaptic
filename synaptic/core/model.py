from datetime import datetime, timezone
from typing import Any, List

from ..providers import DeepSeekAdapter, GeminiAdapter, OpenAIAdapter
from .base import History, ResponseFormat, ResponseMem, UserMem
from .provider import Provider
from .tool import Tool, ToolCall


class Model:
    """
    Universal model which will allow usage of different LLM providers.
    """

    def __init__(
        self,
        provider: Provider,
        model: str,
        temperature: float = 0.8,
        api_key: str = "",  # type: ignore
        max_tokens: int = 1024,
        tools: List[Tool] = None,  # type: ignore
        history: History = History(),
        autorun: bool = False,
        automem: bool = False,
        blacklist: List[str] = [],  # type: ignore
        instructions: str = "",
        response_format: ResponseFormat = ResponseFormat.NONE,  # type: ignore
        response_schema: Any = None,  # type: ignore
    ) -> None:
        """
        Constructor:
        provider:
            type: Provider -- File: synaptic.core.provider Class Provider
            description: enum
                values:
                    - OPENAI
                    - GEMINI
        model:
            type: str
            description: model name to use from the provider
        temperature:
            type: float
            description: sampling temperature for response generation
        api_key:
            type: str
            description: API key for the provider
        max_tokens:
            type: int
            description: maximum tokens for the response
        tools:
            type: List[Tool] -- File: synaptic.core.tool Class Tool
            description: list of tools to bind to the model
        history:
            type: History -- File: synaptic.core.base.memory Class History
            description: conversation history manager
        autorun:
            type:bool
            description: whether to automatically run tools returned by the model
        automem:
            type: bool
            description: whether to automatically store interactions in history
        blkacklist:
            type: List[str]
            description: list of tool names to ignore in autorun
        """
        # set attribute values
        self.provider = provider
        self.model = model
        self.temperature = temperature
        if api_key == "":
            raise ValueError("API key must be provided")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.tools = tools or None
        self.autorun = autorun
        self.automem = automem
        self.history: History = history if self.automem else None  # type: ignore
        self.blacklist = blacklist
        self.response_format = response_format
        self.response_schema = response_schema
        if self.response_format != ResponseFormat.NONE and self.response_schema is None:
            raise ValueError(
                "Response schema must be provided for sturctured response formats"
            )
        self.llm = self._initiate_model()
        self.llm._invalidate_tools()
        self.tools = self.llm.synaptic_tools
        self.instructions = instructions

    def bind_tools(self, tools: List[Tool]) -> None:
        """
        bind_tools:
            type: method
            return type: None
            parameters:
                - List[Tool] -- File: synaptic.core.tool Class Tool
            description: bind additional tools to the model
        """
        # append the tools
        if self.tools is None:
            self.tools = []
        self.tools += tools
        self.llm.synaptic_tools = self.tools
        self.llm._invalidate_tools()

    def _initiate_model(self) -> Any:
        """
        _initiate_model:
            type: method
            return type: BaseModel -- File: synaptic.core.base.base_model Class BaseModel
            parameters: None
            description: initiate the model based on the provider
        """
        # initialize the model based on the provider
        if self.provider == Provider.OPENAI:
            return OpenAIAdapter(
                model=self.model,
                temperature=self.temperature,
                tools=(
                    self.tools if self.response_format == ResponseFormat.NONE else None
                ),
                history=self.history,
                api_key=self.api_key,
                response_format=self.response_format,
                response_schema=self.response_schema,
            )
        elif self.provider == Provider.GEMINI:
            return GeminiAdapter(
                model=self.model,
                temperature=self.temperature,
                tools=(
                    self.tools if self.response_format == ResponseFormat.NONE else None
                ),
                history=self.history,
                api_key=self.api_key,
                response_format=self.response_format,
                response_schema=self.response_schema,
            )
        elif self.provider == Provider.DEEPSEEK:
            return DeepSeekAdapter(
                model=self.model,
                temperature=self.temperature,
                tools=(
                    self.tools if self.response_format == ResponseFormat.NONE else None
                ),
                history=self.history,
                api_key=self.api_key,
                response_format=self.response_format,
                response_schema=self.response_schema,
            )
        # default to Gemini if unknown
        else:
            return GeminiAdapter(
                model=self.model,
                temperature=self.temperature,
                tools=(
                    self.tools if self.response_format == ResponseFormat.NONE else None
                ),
                history=self.history,
                api_key=self.api_key,
                response_format=self.response_format,
                response_schema=self.response_schema,
            )

    def _run_tools(self, tool_calls: List[ToolCall]) -> List[Any]:
        """
        run_tools:
            type: method
            return type: List[Any]
            parameters:
                - List[ToolCall] -- File: synaptic.core.tool Class ToolCall
            description: internal tool runner, requires autorun to be true
        """
        # prepare a results list
        results = []
        if not hasattr(self.llm, "synaptic_tools") or not self.llm.synaptic_tools:
            return results
        if self.tools is None or len(self.tools) == 0:
            return results
        # get registered tools
        tool_map = {tool.name: tool.function for tool in self.llm.synaptic_tools}

        for call in tool_calls:
            name, args = call.name, call.args
            if name in tool_map and name not in self.blacklist:
                try:
                    results.append({"name": name, "result": tool_map[name](**args)})
                except Exception as e:
                    results.append({"name": name, "error": str(e)})
            else:
                results.append(
                    {"name": name, "error": "Tool not registered or blacklisted"}
                )
        return results

    def invoke(
        self, prompt: str, role: str = "user", autorun: bool = None, automem: bool = None, **kwargs  # type: ignore
    ) -> ResponseMem:
        """
        invoke:
            type: method
            return type: ResponseMem -- File: synaptic.core.base.memory Class ResponseMem
            parameters:
                - prompt: str
                - autorun: bool (optional, overrides instance autorun)
                - automem: bool (optional, overrides instance automem)
                - **kwargs: additional parameters for the model's invoke method
            description: invoke the model with a prompt optionally, optionally provide role, auto run tools and auto manage memory
        """
        # verify role
        if role not in ["user", "assistant", "system"]:
            raise ValueError("Role must be one of 'user', 'assistant', or 'system'")

        created = (datetime.now().astimezone(timezone.utc),)

        # send to provider specific invokation
        memory = self.llm.invoke(prompt=prompt, role=role, **kwargs)

        # do priority check for autorun and automem -- in call > in object
        autorun = autorun if (autorun is not None) else self.autorun
        automem = automem if (automem is not None) else self.automem

        # run auto management
        if autorun:
            if memory.tool_calls:
                tool_results = self._run_tools(memory.tool_calls)
                # Attach results for downstream use
                memory.tool_results = tool_results
        else:
            memory.tool_results = []

        if automem:
            self.history.add(UserMem(message=prompt, role=role, created=created))
            self.history.add(memory)

        return memory
