from datetime import datetime, timezone
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

from ...core.base import BaseModel, ResponseMem, History

load_dotenv()


class GeminiAdapter(BaseModel):
    def __init__(self, model: str, tools: list | None = None):
        self.client = genai.Client()
        self.model = model
        self.tools = tools or []

    def _convert_tools(self) -> list[types.Tool]:
        """Convert custom Tool objects to Gemini `types.Tool` objects."""
        gemini_tools = []
        for t in self.tools:
            # Each Tool may have a declaration dict
            gemini_tools.append(types.Tool(function_declarations=[t.declaration]))
        return gemini_tools

    def invoke(self, prompt: str, **kwargs) -> ResponseMem:
        # Build config with tools if any
        tools = self._convert_tools()
        config = types.GenerateContentConfig(tools=tools) if self.tools else None

        # Wrap prompt into `types.Content`
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

        # Call Gemini
        response = self.client.models.generate_content(
            model=self.model, contents=contents, config=config, **kwargs
        )

        # Extract metadata
        created = datetime.now().astimezone(timezone.utc)
        message = ""
        tool_calls = []

        if response.candidates:
            candidate = response.candidates[0]

            # Only process content if it exists
            if candidate.content:
                for part in candidate.content.parts:  # type: ignore
                    if part.text:
                        message += part.text
                    if part.function_call:
                        tool_calls.append(
                            {
                                "name": part.function_call.name,
                                "args": part.function_call.args,
                            }
                        )

        return ResponseMem(message=message, created=created, tool_calls=tool_calls)
