import inspect
import types
from typing import Any, Callable, Dict, List, Optional


class ToolRegistry:
    """Collection of tools for one model, agent, session, or application."""

    def __init__(self) -> None:
        self._tools: Dict[str, "Tool"] = {}
        self._callbacks: List[Callable[[], None]] = []

    def register(self, tool: "Tool") -> None:
        """Add or replace a tool by name and notify adapter subscribers."""
        self._tools[tool.name] = tool
        self._notify()

    def register_many(self, tools: List["Tool"]) -> None:
        for tool in tools:
            self._tools[tool.name] = tool
        if tools:
            self._notify()

    def unregister(self, name: str) -> None:
        if name in self._tools:
            del self._tools[name]
            self._notify()

    def unregister_prefix(self, prefix: str) -> None:
        names = [name for name in self._tools if name.startswith(prefix)]
        for name in names:
            del self._tools[name]
        if names:
            self._notify()

    def _notify(self) -> None:
        for cb in self._callbacks:
            cb()

    def on_change(self, fn: Callable[[], None]) -> None:
        self._callbacks.append(fn)

    def all(self) -> Dict[str, "Tool"]:
        return self._tools

    def autotool(
        self,
        description: str = "",
        param_descriptions: Optional[Dict[str, str]] = None,
        default_params: Optional[dict] = None,
    ) -> Callable:
        return autotool(
            description=description,
            param_descriptions=param_descriptions,
            default_params=default_params,
            registry=self,
        )


_default_registry = ToolRegistry()
TOOL_REGISTRY: Dict[str, "Tool"] = _default_registry._tools


def register_callback(fn: Callable[[], None], registry: Optional[ToolRegistry] = None) -> None:
    """Subscribe to changes on a registry, defaulting to the process-wide one."""
    (registry or _default_registry).on_change(fn)


def collect_tools(
    tools: Optional[List["Tool"]] = None,
    registry: Optional[ToolRegistry] = None,
) -> Dict[str, "Tool"]:
    """Merge explicit tools with a registry, preferring explicit tools."""
    all_tools: Dict[str, "Tool"] = {}
    for tool in tools or []:
        all_tools[tool.name] = tool
    source = registry.all() if registry is not None else TOOL_REGISTRY
    for name, tool in source.items():
        if name not in all_tools:
            all_tools[name] = tool
    return all_tools


class Tool:
    """Callable exposed to model providers through function/tool calling."""

    def __init__(
        self,
        name: str,
        declaration: dict,
        function: Callable[..., Any] = lambda: None,
        default_params: Optional[dict] = None,
        add_to_registry: bool = True,
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.name = name
        self.declaration = declaration
        if not callable(function):
            raise ValueError("Function must be a callable")
        self.function = function
        self.is_async = inspect.iscoroutinefunction(function)
        self.default_params = default_params or {}

        if add_to_registry:
            target = registry or _default_registry
            target.register(self)

        self.run = self._run_async if self.is_async else self._run_sync

    def __get__(self, instance, owner):
        """Bind decorated methods to their instance without re-registering."""
        if instance is None:
            return self
        return Tool(
            name=self.name,
            declaration=self.declaration,
            function=types.MethodType(self.function, instance),
            default_params=self.default_params,
            add_to_registry=False,
        )

    def _run_sync(self, **kwargs):
        """Run a sync tool with defaults merged before call-time arguments."""
        return self.function(**{**self.default_params, **kwargs})

    async def _run_async(self, **kwargs):
        """Run an async tool with defaults merged before call-time arguments."""
        return await self.function(**{**self.default_params, **kwargs})


class ToolCall:
    """Parsed model request to call a named tool with JSON-like arguments."""

    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args

    def get_arg(self, key: str):
        return self.args.get(key)

    def list_args(self):
        return self.args.keys()

    def __repr__(self):
        short = {k: v for k, v in self.args.items() if len(str(v)) < 50}
        return f"<ToolCall name={self.name!r} args={short}>"


def autotool(
    description: str = "",
    param_descriptions: Optional[Dict[str, str]] = None,
    default_params: Optional[dict] = None,
    autobind: bool = True,
    registry: Optional[ToolRegistry] = None,
) -> Callable:
    """Decorate a function as a `Tool` with an inferred parameter schema."""
    param_descriptions = param_descriptions or {}

    def decorator(func: Callable[..., Any]) -> Tool:
        sig = inspect.signature(func)
        properties: dict = {}
        required: list = []

        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            anno = param.annotation
            ptype = (
                "number" if anno is float
                else "integer" if anno is int
                else "boolean" if anno is bool
                else "string"
            )
            properties[pname] = {"type": ptype, "description": param_descriptions.get(pname, "")}
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        parameters = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required

        return Tool(
            name=func.__name__,
            declaration={"name": func.__name__, "description": description, "parameters": parameters},
            function=func,
            default_params=default_params,
            add_to_registry=autobind,
            registry=registry,
        )

    return decorator
