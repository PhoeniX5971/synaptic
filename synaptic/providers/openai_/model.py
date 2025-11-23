import json
from datetime import datetime, timezone
from typing import Any, List

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from ...core.base import BaseModel, History, ResponseFormat, ResponseMem
from ...core.tool import TOOL_REGISTRY, ToolCall, register_callback

load_dotenv()


class OpenAIAdapter(BaseModel):

    def __init__(
        self,
        model: str,
        history: History,
        api_key: str,
        response_format: ResponseFormat,
        response_schema: Any,
        temperature: float = 0.8,
        tools: list | None = None,
        instructions: str = "",  # ← NEW
    ):

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.synaptic_tools = tools
        self.openai_tools = []
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.instructions = instructions  # ← NEW
        register_callback(self._invalidate_tools)
        self._invalidate_tools()
        self.role_map = {
            "user": "user",
            "assistant": "assistant",
            "system": "system",
        }

    def _invalidate_tools(self):
        self._convert_tools()

    def _convert_tools(self):
        self.openai_tools = []

        # if self.response_format != ResponseFormat.NONE:
        #     return None

        all_tools = {}

        for t in self.synaptic_tools or []:
            all_tools[t.name] = t

        for t_name, t in TOOL_REGISTRY.items():
            if t_name not in all_tools:
                all_tools[t_name] = t

        for t_name, t in all_tools.items():
            self.openai_tools.append(
                {
                    "name": t.name,
                    "description": t.declaration.get("description", ""),
                    "parameters": t.declaration.get("parameters", {}),
                }
            )

        self.synaptic_tools = list(all_tools.values())

    def to_messages(self) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []

        for memory in self.history.MemoryList:
            content_text = memory.message + f"\n(Created at: {memory.created})"

            if isinstance(memory, ResponseMem):
                if memory.tool_calls:
                    content_text += f"\nTool calls: {memory.tool_calls}"
                if getattr(memory, "tool_results", []):
                    content_text += f"\nTool results: {memory.tool_results}"

            role = self.role_map.get(memory.role, "user")

            if role == "user":
                msg = ChatCompletionUserMessageParam(content=content_text, role="user")
            elif role == "system":
                msg = ChatCompletionSystemMessageParam(
                    content=content_text, role="system"
                )
            elif role == "assistant":
                msg = ChatCompletionAssistantMessageParam(
                    content=content_text, role="assistant"
                )
            else:
                msg = ChatCompletionUserMessageParam(content=content_text, role="user")

            messages.append(msg)

        return messages

    def invoke(self, prompt: str, role: str = "user", **kwargs) -> ResponseMem:
        messages = self.to_messages()

        # ★ PREPEND SYSTEM INSTRUCTIONS ★
        if self.instructions:
            messages.insert(
                0,
                ChatCompletionSystemMessageParam(
                    content=self.instructions, role="system"
                ),
            )

        # Add user/system/assistant message for this prompt
        if role == "user":
            message = ChatCompletionUserMessageParam(content=prompt, role="user")
        elif role == "assistant":
            message = ChatCompletionAssistantMessageParam(
                content=prompt, role="assistant"
            )
        elif role == "system":
            message = ChatCompletionSystemMessageParam(content=prompt, role="system")
        else:
            message = ChatCompletionUserMessageParam(content=prompt, role="user")

        messages.append(message)

        params = {
            "model": self.model,
            "messages": messages,
            "functions": self.openai_tools,
            **kwargs,
        }

        # JSON schema mode
        if self.response_format == ResponseFormat.JSON:
            params["response_format"] = self.response_schema
            params["functions"] = None

        # Normal or JSON-mode call
        response = (
            self.client.chat.completions.parse(**params)
            if self.response_format == ResponseFormat.JSON
            else self.client.chat.completions.create(**params)
        )

        created = datetime.now().astimezone(timezone.utc)

        # Extract content from all choices
        message_texts = []
        tool_calls: List[ToolCall] = []

        for choice in response.choices:
            msg = choice.message
            if msg:
                # Collect content
                if msg.content:
                    message_texts.append(msg.content)

                # Collect tool calls
                if msg.function_call:
                    fc = msg.function_call
                    tool_calls.append(
                        ToolCall(name=fc.name, args=json.loads(fc.arguments))
                    )

        message_text = (
            "\n".join(message_texts) if message_texts else "No content available"
        )

        return ResponseMem(message=message_text, created=created, tool_calls=tool_calls)
