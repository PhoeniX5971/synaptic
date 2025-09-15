import json
from datetime import datetime, timezone
from typing import List

from dotenv import load_dotenv

from ...core.base import BaseModel, ResponseMem
from ...core.tool import Tool

load_dotenv()


class OpenAIAdapter(BaseModel):
    def __init__(self, model: str, tools: List[Tool] | None = None):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.tools = tools or []

    def _convert_tools(self):
        """Convert internal Tool objects to OpenAI function definitions"""
        functions = []
        for t in self.tools:
            functions.append(
                {
                    "name": t.name,
                    "description": t.declaration.get("description", ""),
                    "parameters": t.declaration.get("parameters", {}),
                }
            )
        return functions

    def invoke(self, prompt: str, **kwargs) -> ResponseMem:
        # Call OpenAI Chat Completions
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            functions=self._convert_tools() if self.tools else None,  # type: ignore
            **kwargs,
        )

        created = datetime.now().astimezone(timezone.utc)
        choice = response.choices[0]

        # Extract message content
        message = choice.message.get("content") if choice.message else ""

        # Extract tool calls if any
        tool_calls = []
        if choice.message and choice.message.get("function_call"):
            fc = choice.message["function_call"]
            tool_calls.append({"name": fc["name"], "args": json.loads(fc["arguments"])})

        return ResponseMem(message=message, created=created, tool_calls=tool_calls)
