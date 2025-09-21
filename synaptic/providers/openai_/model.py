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
from ...core.tool import ToolCall

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
    ):

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tools = tools
        self.temperature = temperature
        self.history = history
        self.response_format = response_format
        self.response_schema = response_schema
        self.role_map = {
            "user": "user",
            "assistant": "assistant",
            "system": "system",
        }

    def _convert_tools(self):
        """Convert internal Tool objects to OpenAI function definitions"""
        functions = []
        if self.tools is None:
            return None
        for t in self.tools:
            functions.append(
                {
                    "name": t.name,
                    "description": t.declaration.get("description", ""),
                    "parameters": t.declaration.get("parameters", {}),
                }
            )
        return functions

    def to_messages(self) -> list[ChatCompletionMessageParam]:
        """Convert all memories to OpenAI ChatCompletion message objects."""
        messages: list[ChatCompletionMessageParam] = []

        for memory in self.history.MemoryList:
            # Build base content
            content_text = memory.message + f"\n(Created at: {memory.created})"

            # Include tool calls / results inline if ResponseMem
            if isinstance(memory, ResponseMem):
                if memory.tool_calls:
                    content_text += f"\nTool calls: {memory.tool_calls}"
                if getattr(memory, "tool_results", []):
                    content_text += f"\nTool results: {memory.tool_results}"

            # Map role to correct OpenAI message type
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
                # Fallback to user
                msg = ChatCompletionUserMessageParam(content=content_text, role="user")

            messages.append(msg)

        return messages

    def invoke(self, prompt: str, role: str = "user", **kwargs) -> ResponseMem:
        tools = self._convert_tools()

        messages = self.to_messages()
        # Choose the right message param class based on role
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
            "functions": tools,
            **kwargs,
        }

        if self.response_format == ResponseFormat.JSON:
            params["response_format"] = self.response_schema

        response = (
            self.client.chat.completions.parse(**params)
            if self.response_format == ResponseFormat.JSON
            else self.client.chat.completions.create(**params)
        )

        created = datetime.now().astimezone(timezone.utc)
        choice = response.choices[0]

        # Extract message content
        message = (
            choice.message.content
            if choice.message and choice.message.content
            else "No content available"
        )

        # Extract tool calls if any
        tool_calls: List[ToolCall] = []
        if choice.message and choice.message.function_call:
            fc = choice.message.function_call
            tool_calls.append(
                ToolCall(
                    name=fc.name,
                    args=json.loads(fc.arguments),
                )
            )

        return ResponseMem(message=message, created=created, tool_calls=tool_calls)
