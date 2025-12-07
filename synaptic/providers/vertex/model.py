from datetime import datetime, timezone
from typing import Any, List, Optional

from dotenv import load_dotenv

from vertexai import init as vertex_init
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    Content,
    Tool,
    GenerationConfig,
    FunctionDeclaration,
)

from ...core.base import BaseModel, History, ResponseFormat, ResponseMem
from ...core.tool import TOOL_REGISTRY, ToolCall, register_callback, Tool as ST


load_dotenv()


class VertexAdapter(BaseModel):
    def __init__(
        self,
        model: str,
        project: str,
        location: str,
        history: History | None,
        response_format: ResponseFormat,
        response_schema: Any,
        tools: Optional[List[ST]],
        api_key: str | None = None,
        temperature: float = 0.8,
        instructions: str = "",
    ):
        vertex_init(project=project, location=location)

        self.model_name = model
        self.model = GenerativeModel(model)
        self.temperature = temperature
        self.history = history
        self.synaptic_tools = list(tools or [])
        self.vertex_tools: List[Tool] = []
        self.instructions = instructions
        self.response_format = response_format
        self.response_schema = response_schema

        self.role_map = {
            "user": "user",
            "assistant": "model",
            "system": "user",
        }

        register_callback(self._invalidate_tools)
        self._invalidate_tools()

    # ----------------------
    # Tool Conversion
    # ----------------------
    def _invalidate_tools(self):
        self._convert_tools()

    def _convert_tools(self):
        """Convert TOOL_REGISTRY + explicit tools → Vertex Tool definitions."""
        all_tools = {}
        all_declarations: List[FunctionDeclaration] = (
            []
        )  # <-- New list for declarations

        # merge local tools + registry
        for t in self.synaptic_tools:
            all_tools[t.name] = t
        for name, t in TOOL_REGISTRY.items():
            if name not in all_tools:
                all_tools[name] = t

        for _, tool in all_tools.items():
            decl = tool.declaration

            if isinstance(decl, dict):
                decl = FunctionDeclaration(
                    name=decl.get("name"),  # type: ignore
                    description=decl.get("description"),  # type: ignore
                    parameters=decl.get("parameters"),  # type: ignore
                )

            # Add the declaration to the consolidated list
            all_declarations.append(decl)

        # ----------------------------------------------------
        # The Fix: Wrap all declarations in a single Tool object
        # ----------------------------------------------------
        if all_declarations:
            # `self.vertex_tools` will now contain a list with only ONE Tool object
            self.vertex_tools = [Tool(function_declarations=all_declarations)]
        else:
            self.vertex_tools = []  # handle case with no tools

        self.synaptic_tools = list(all_tools.values())

    # ----------------------
    # History → Vertex content
    # ----------------------
    def to_contents(self) -> List[Content]:
        contents = []

        if self.history is None:
            return contents
        for mem in self.history.MemoryList:
            parts = [Part.from_text(mem.message)]
            parts.append(Part.from_text(f"(Created at: {mem.created})"))

            if isinstance(mem, ResponseMem):
                if mem.tool_calls:
                    parts.append(Part.from_text(f"Tool calls: {mem.tool_calls}"))
                if getattr(mem, "tool_results", []):
                    parts.append(Part.from_text(f"Tool results: {mem.tool_results}"))

            contents.append(
                Content(
                    role=self.role_map.get(mem.role, "user"),
                    parts=parts,
                )
            )

        return contents

    # ----------------------
    # Main Invoke
    # ----------------------
    def invoke(self, prompt: str, role: str = "user", **kwargs) -> ResponseMem:
        role = self.role_map.get(role, "user")

        history_contents = self.to_contents()
        user_message = Content(role=role, parts=[Part.from_text(prompt)])

        messages: List[Content] = history_contents + [user_message]

        if self.instructions:
            system_msg = Content(role="user", parts=[Part.from_text(self.instructions)])
            messages = [system_msg] + messages

        # ----------------------
        # Config
        # ----------------------
        if self.response_format == ResponseFormat.NONE:
            response_mime = "text/plain"
        elif self.response_format == ResponseFormat.JSON:
            response_mime = "application/json"
        else:
            response_mime = "text/plain"

        config = GenerationConfig(
            temperature=self.temperature,
            response_mime_type=response_mime,
        )

        # ----------------------
        # Call Vertex AI
        # ----------------------
        response = self.model.generate_content(
            messages,
            generation_config=config,
            tools=self.vertex_tools,
        )

        created = datetime.now().astimezone(timezone.utc)
        message = ""
        tool_calls: List[ToolCall] = []

        # ----------------------
        # Parse Vertex response
        # ----------------------
        if response.candidates:
            cand = response.candidates[0]

            # 1. Safely check for function calls directly from the candidate
            if cand.function_calls:  # This is a direct list provided by the SDK
                for fc in cand.function_calls:
                    tool_calls.append(
                        ToolCall(
                            name=fc.name,
                            args=dict(fc.args) or {},  # Ensure args is a dict
                        )
                    )

            # 2. Extract text from the parts list
            if cand.content and cand.content.parts:
                for p in cand.content.parts:
                    if p.text:
                        message += p.text

            # 3. Handle the case where the model only returns a function call (message is still "")
            if tool_calls and not message:
                # Set a non-error state for the message if the model only requested a tool.
                message = ""

        return ResponseMem(
            message=message,
            created=created,
            tool_calls=tool_calls,
        )
