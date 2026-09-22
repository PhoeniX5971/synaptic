"""Explicit bindings and dynamic registry-derived tool snapshots."""

from typing import List
from .tool import Tool, collect_tools


from .base import BaseModel


class ModelBindings:
    llm: BaseModel

    @property
    def tools(self) -> List[Tool]:
        """Current adapter tool snapshot, including registry changes."""
        return list(self.llm.synaptic_tools or [])

    @tools.setter
    def tools(self, tools: List[Tool]) -> None:
        # Retain assignment compatibility; prefer bind_tools/unbind_tools.
        if hasattr(self, "llm"):
            self.llm.synaptic_tools = list(tools or [])

    def bind_tools(self, tools: List[Tool], *, replace: bool = False) -> None:
        """Bind unique names; reject conflicting definitions unless replace=True."""
        bound = {t.name: t for t in self.llm.bound_tools or []}
        for tool in tools:
            existing = bound.get(tool.name)
            if existing is not None and existing is not tool and not replace:
                raise ValueError(f"Tool {tool.name!r} is already bound; use replace=True")
            bound[tool.name] = tool
        self._set_bound_tools(list(bound.values()))

    def unbind_tools(self, *names: str) -> None:
        """Remove explicit bindings; registry tools remain registry-owned."""
        self._set_bound_tools([t for t in self.llm.bound_tools if t.name not in names])

    def _set_bound_tools(self, tools: List[Tool]) -> None:
        self.llm.bound_tools = list(tools)
        registry = getattr(self, "tool_registry", None)
        self.llm.synaptic_tools = list(collect_tools(tools, registry).values())
        self.llm._invalidate_tools()

    # ------------------------------------------------------------------ #
    # Invoke / stream delegation                                           #
    # ------------------------------------------------------------------ #
