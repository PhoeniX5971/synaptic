"""A tool whose valid call ends the run with a structured result."""

from typing import Any, Callable, List, Optional

from ..core.tool import Tool
from ._extend import Extension


class FinalAnswer(Extension):
    """Offer `name` as a run-ending tool whose arguments follow `schema`.

    `schema` is a Pydantic-style class or a JSON-schema dict. With `required=True`
    the run may not end with plain text. Several can be used with different names;
    the first valid call ends the run.
    """

    def __init__(
        self,
        schema: Any,
        name: str = "final_answer",
        description: str = "Give the final result and end the task. Call it alone.",
        required: bool = False,
        allowed: Optional[Callable[[], Optional[str]]] = None,
    ):
        self.schema, self.tool_name, self.required, self.allowed = schema, name, required, allowed
        self.name = "final:" + name
        typed = hasattr(schema, "model_json_schema")
        parameters = schema.model_json_schema() if typed else schema
        self._tool = Tool(
            name,
            {"name": name, "description": description, "parameters": parameters},
            self._submit,
            add_to_registry=False,
            metadata={"errors": "return", "final": True},
        )

    def tools(self) -> List[Tool]:
        return [self._tool]

    def before_finish(self, final: bool = False) -> Optional[str]:
        if final or not self.required:
            return None
        return f"Finish by calling the {self.tool_name} tool."

    def _submit(self, **arguments: Any) -> str:
        agent = self.agent
        refusal = self.allowed() if self.allowed else None
        if refusal:
            raise ValueError(refusal)
        if len(agent.pending_calls) > 1:
            raise ValueError(
                f"Call {self.tool_name} alone, after the results of your other calls are in."
            )
        blocked = agent.blockers(final=True)
        if blocked:
            raise ValueError(" ".join(blocked))
        if hasattr(self.schema, "model_validate"):
            output = self.schema.model_validate(arguments)
            agent.finish(output, self.tool_name, output.model_dump(mode="json"))
        else:
            agent.finish(dict(arguments), self.tool_name)
        return "accepted"
