from datetime import datetime, timezone
from typing import Any, List, Optional
import inspect

from ..providers import DeepSeekAdapter, GeminiAdapter, OpenAIAdapter, VertexAdapter
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
        tools: Optional[List[Tool]] = None,
        history: Optional[History] = None,
        autorun: bool = False,
        automem: bool = False,
        blacklist: List[str] | None = None,
        location: Optional[str] = None,
        project: Optional[str] = None,
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
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.tools = tools or []
        self.autorun = autorun
        self.automem = automem
        self.history = history or History()
        self.blacklist = blacklist or []
        self.response_format = response_format
        self.response_schema = response_schema
        if self.response_format != ResponseFormat.NONE and self.response_schema is None:
            raise ValueError(
                "Response schema must be provided for structured response formats"
            )
        self.location = location
        self.instructions = instructions
        self.project = project
        self.llm = self._initiate_model()
        self.llm._invalidate_tools()
        self.tools = self.llm.synaptic_tools

        if tools:
            self.bind_tools(tools)

        if not self.automem:
            self.history.window(1)

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
        tools = self.tools if self.response_format == ResponseFormat.NONE else None

        if self.provider == Provider.OPENAI:
            return OpenAIAdapter(
                model=self.model,
                temperature=self.temperature,
                tools=tools,
                history=self.history,
                api_key=self.api_key,
                response_format=self.response_format,
                response_schema=self.response_schema,
                instructions=self.instructions,
            )
        elif self.provider == Provider.GEMINI:
            return GeminiAdapter(
                model=self.model,
                temperature=self.temperature,
                tools=tools,
                history=self.history,
                api_key=self.api_key,
                response_format=self.response_format,
                response_schema=self.response_schema,
                instructions=self.instructions,
            )
        elif self.provider == Provider.VERTEX:
            return VertexAdapter(
                model=self.model,
                temperature=self.temperature,
                tools=tools,
                history=self.history,
                api_key=None,
                response_format=self.response_format,
                response_schema=self.response_schema,
                instructions=self.instructions,
                location=self.location,  # type: ignore
                project=self.project,  # type: ignore
            )
        elif self.provider == Provider.DEEPSEEK:
            return DeepSeekAdapter(
                model=self.model,
                temperature=self.temperature,
                tools=tools,
                history=self.history,  # type: ignore
                api_key=self.api_key,
                response_format=self.response_format,
                response_schema=self.response_schema,
            )
        # default to Gemini if unknown
        else:
            return GeminiAdapter(
                model=self.model,
                temperature=self.temperature,
                tools=tools,
                history=self.history,
                api_key=self.api_key,
                response_format=self.response_format,
                response_schema=self.response_schema,
                instructions=self.instructions,
            )

    def _run_tools(self, tool_calls: List[ToolCall]) -> List[Any]:
        results = []
        tool_map = {tool.name: tool for tool in self.llm.synaptic_tools}

        for call in tool_calls:
            if not isinstance(call, ToolCall):
                results.append({"error": "Invalid tool call format"})
                continue
            name, args = call.name, call.args
            tool = tool_map.get(name)

            if tool and name not in self.blacklist:
                if tool.is_async:
                    raise RuntimeError(
                        f"Tool '{name}' is async, cannot run in sync invoke()"
                    )
                try:
                    res = tool.run(**args)
                    results.append({"name": name, "result": res})
                except Exception as e:
                    results.append({"name": name, "error": str(e)})
            else:
                results.append(
                    {"name": name, "error": "Tool not registered or blacklisted"}
                )
        return results

    async def _arun_tools(self, tool_calls: List[ToolCall]) -> List[Any]:
        """
        run_tools:
            type: method
            return type: List[Any]
            parameters:
                - List[ToolCall] -- File: synaptic.core.tool Class ToolCall
            description: internal tool runner, requires autorun to be true
        """
        results = []
        tool_map = {tool.name: tool for tool in self.llm.synaptic_tools}

        for call in tool_calls:
            if not isinstance(call, ToolCall):
                results.append({"error": "Invalid tool call format"})
                continue
            name, args = call.name, call.args
            tool = tool_map.get(name)

            if tool and name not in self.blacklist:
                try:
                    # The magic line:
                    res = tool.run(**args)
                    if inspect.iscoroutine(res):
                        res = await res

                    results.append({"name": name, "result": res})
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

        created = datetime.now().astimezone(timezone.utc)

        # send to provider specific invokation
        memory = self.llm.invoke(prompt=prompt, role=role, **kwargs)

        # do priority check for autorun and automem -- in call > in object
        autorun = autorun if (autorun is not None) else self.autorun
        automem = automem if (automem is not None) else self.automem

        # run auto management
        if autorun:
            if any(tool.is_async for tool in self.llm.synaptic_tools):
                raise RuntimeError("invoke() cannot run async tools; use ainvoke()")
            if memory.tool_calls:
                tool_results = self._run_tools(memory.tool_calls)
                # Attach results for downstream use
                memory.tool_results = tool_results
        else:
            memory.tool_results = []

        if automem and self.history:
            self.history.add(UserMem(message=prompt, role=role, created=created))
            self.history.add(memory)

        return memory

    async def ainvoke(
        self, prompt: str, role: str = "user", autorun: bool = None, automem: bool = None, **kwargs  # type: ignore
    ) -> ResponseMem:
        """
        ainvoke:
            type: method
            return type: ResponseMem -- File: synaptic.core.base.memory Class ResponseMem
            parameters:
                - prompt: str
                - autorun: bool (optional, overrides instance autorun)
                - automem: bool (optional, overrides instance automem)
                - **kwargs: additional parameters for the model's invoke method
            description: same as invoke but async, utilizes a different async runner
        """
        # verify role
        if role not in ["user", "assistant", "system"]:
            raise ValueError("Role must be one of 'user', 'assistant', or 'system'")

        created = datetime.now().astimezone(timezone.utc)

        # send to provider specific invokation
        memory = self.llm.invoke(prompt=prompt, role=role, **kwargs)

        # do priority check for autorun and automem -- in call > in object
        autorun = autorun if (autorun is not None) else self.autorun
        automem = automem if (automem is not None) else self.automem

        # run auto management
        if autorun:
            if memory.tool_calls:
                tool_results = await self._arun_tools(memory.tool_calls)
                # Attach results for downstream use
                memory.tool_results = tool_results
        else:
            memory.tool_results = []

        if automem and self.history:
            self.history.add(UserMem(message=prompt, role=role, created=created))
            self.history.add(memory)

        return memory

    async def astream(
        self, prompt: str, role: str = "user", autorun: bool = None, automem: bool = None, **kwargs  # type: ignore
    ):
        """
        astream:
            Async generator wrapper around provider streaming (llm.astream).
            - Yields raw chunks from the provider's astream().
            - If autorun=True, runs tool calls as they appear (using _arun_tools).
            - If automem=True, saves the user prompt and final ResponseMem into history.
            - Produces the same chunk objects the provider yields (no transformation).
        """
        # verify role
        if role not in ["user", "assistant", "system"]:
            raise ValueError("Role must be one of 'user', 'assistant', or 'system'")

        # priority: argument > instance setting
        autorun = autorun if (autorun is not None) else self.autorun
        automem = automem if (automem is not None) else self.automem

        # ensure the underlying LLM supports streaming
        if not hasattr(self.llm, "astream"):
            raise NotImplementedError("Underlying model does not implement astream()")

        created = datetime.now().astimezone(timezone.utc)

        accumulated_message = ""
        tool_calls: List[ToolCall] = []
        tool_results: List[Any] = []

        # Iterate provider async stream and yield chunks downstream
        async for chunk in self.llm.astream(prompt=prompt, role=role, **kwargs):
            # Yield the chunk as-is so callers receive it in real-time
            yield chunk

            # Collect text pieces if they exist
            if getattr(chunk, "text", None):
                accumulated_message += chunk.text

            # If the chunk contains a function call, collect and optionally run it immediately
            if getattr(chunk, "function_call", None):
                # Record the call
                tool_calls.append(chunk.function_call)

                # If autorun, execute this call now (supports async and sync tools)
                if autorun and chunk.function_call:
                    try:
                        # _arun_tools expects a list
                        tr = await self._arun_tools([chunk.function_call])
                        tool_results.extend(tr)
                    except Exception as e:
                        # keep streaming even if a tool fails; append the error to results
                        tool_results.append({"name": chunk.function_call.name, "error": str(e)})

        # Stream finished — assemble the final ResponseMem and attach tool results
        final_mem = ResponseMem(message=accumulated_message, created=created, tool_calls=tool_calls)
        final_mem.tool_results = tool_results

        # If automem, persist UserMem + ResponseMem
        if automem and self.history:
            self.history.add(UserMem(message=prompt, role=role, created=created))
            self.history.add(final_mem)

        # NOTE: async generators can't "return" a value usefully; the final memory
        # is available in history (or in 'final_mem' if you adapt this function to expose it).
        # If you want a programmatic handle, callers can inspect `model.history.MemoryList[-1]`.
        return
