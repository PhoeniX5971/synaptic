import inspect
from openai import AsyncOpenAI, OpenAI

from ...core.base import BaseModel, ResponseFormat
from ...core.capabilities import Capabilities
from ...core.tool import collect_tools, register_callback
from .._chat import assemble, response_memory
from .._chat_history import history_messages
from .._media import chat_content


class OpenAIAdapter(BaseModel):
    capabilities = Capabilities(images=True, audio=True)

    @staticmethod
    def _clients(api_key, base_url):
        return (
            OpenAI(api_key=api_key or "none", base_url=base_url),
            AsyncOpenAI(api_key=api_key or "none", base_url=base_url),
        )

    def __init__(
        self,
        model,
        history,
        api_key,
        response_format,
        response_schema=None,
        tools=None,
        temperature=0.8,
        instructions="",
        tool_registry=None,
        max_tokens=1024,
        base_url=None,
        **kwargs,
    ):
        self.client, self.async_client = self._clients(api_key, base_url)
        self.model, self.history = model, history
        self.temperature, self.instructions = temperature, instructions
        self.response_format, self.response_schema = response_format, response_schema
        self.max_tokens, self.tool_registry = max_tokens, tool_registry
        self.bound_tools = list(tools or [])
        self.role_map = {"user": "user", "assistant": "assistant", "system": "system"}
        self._subscription = register_callback(self._invalidate_tools, tool_registry)
        self._invalidate_tools()

    def _invalidate_tools(self):
        tools = collect_tools(self.bound_tools, self.tool_registry)
        self.synaptic_tools = list(tools.values())
        self.openai_tools = [
            {"type": "function", "function": tool.declaration} for tool in tools.values()
        ]

    def _messages(self, prompt, role, audio=None, images=None):
        messages = history_messages(self.history, self.instructions, self.role_map)
        if prompt is not None or images or audio:
            messages.append({"role": role, "content": chat_content(prompt, images, audio)})
        return messages

    def _request(self, messages, streaming, kwargs):
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if streaming:
            params.update(stream=True, stream_options={"include_usage": True})
        if self.response_format == ResponseFormat.JSON:
            schema = self.response_schema
            if hasattr(schema, "model_json_schema"):
                schema = schema.model_json_schema()
            if schema and self.__class__ is OpenAIAdapter:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": schema},
                }
            else:
                params["response_format"] = {"type": "json_object"}
        elif self.openai_tools:
            params.update(tools=self.openai_tools, tool_choice="auto")
        return params

    def invoke(self, prompt, role="user", images=None, audio=None, **kwargs):
        self.capabilities.check("invoke", images=images, audio=audio)
        response = self.client.chat.completions.create(
            **self._request(self._messages(prompt, role, audio, images), False, kwargs)
        )
        return response_memory(response)

    async def astream(self, prompt, role="user", images=None, audio=None, abort=None, **kwargs):
        self.capabilities.check("astream", images=images, audio=audio)
        source = await self.async_client.chat.completions.create(
            **self._request(self._messages(prompt, role, audio, images), True, kwargs)
        )
        source = assemble(source, abort)
        try:
            async for chunk in source:
                yield chunk
        finally:
            close = getattr(source, "aclose", None) or getattr(source, "close", None)
            if close:
                result = close()
                if inspect.isawaitable(result):
                    await result

    def to_messages(self):
        return history_messages(self.history, self.instructions, self.role_map)

    to_contents = to_messages
