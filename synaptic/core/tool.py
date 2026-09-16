import inspect
import types
from typing import Any, Callable, Dict, List, Optional


from ._registry import ToolRegistry, Subscription
from ._schema import infer, effective
from .invocation import Invocation, freeze
import copy
from uuid import uuid4


_default_registry = ToolRegistry()
TOOL_REGISTRY: Dict[str, "Tool"] = _default_registry._tools


def register_callback(
    fn: Callable[[], None], registry: Optional[ToolRegistry] = None
) -> Subscription:
    """Subscribe to changes on a registry, defaulting to the process-wide one."""
    return (registry or _default_registry).on_change(fn)


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
        input_model=None,
        output_adapter=None,
        metadata=None,
        identity=None,
    ) -> None:
        self.name = name
        self._declaration = copy.deepcopy(declaration)
        self.input_model, self.output_adapter = input_model, output_adapter
        self._metadata, self.identity = copy.deepcopy(metadata or {}), identity
        if not callable(function):
            raise ValueError("Function must be a callable")
        self.function = function
        self.is_async = inspect.iscoroutinefunction(function)
        self._default_params = copy.deepcopy(default_params or {})

        self.run = self._run_async if self.is_async else self._run_sync
        self._sealed = True
        if add_to_registry:
            target = registry or _default_registry
            target.register(self)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Tool definitions are immutable; replace the registration")
        object.__setattr__(self, name, value)

    @property
    def declaration(self):
        return copy.deepcopy(self._declaration)

    @property
    def default_params(self):
        return copy.deepcopy(self._default_params)

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def prepare(self, call):
        values, json_values = effective(
            self.function,
            self._default_params,
            call.args,
            self._declaration.get("parameters", {}),
            self.input_model,
        )
        return Invocation(
            self.name,
            call.id,
            freeze(json_values),
            self.identity,
            self.function,
            values,
            self.output_adapter,
        )

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
            input_model=self.input_model,
            output_adapter=self.output_adapter,
            metadata=self.metadata,
            identity=self.identity,
        )

    def _run_sync(self, **kwargs):
        """Run a sync tool with defaults merged before call-time arguments."""
        invocation = self.prepare(ToolCall(self.name, kwargs))
        value = invocation.execute()
        from ._awaitables import require_sync

        value = require_sync(value, "Async tool result requires async execution")
        return invocation.validate_output(value)

    async def _run_async(self, **kwargs):
        """Run an async tool with defaults merged before call-time arguments."""
        invocation = self.prepare(ToolCall(self.name, kwargs))
        return invocation.validate_output(await invocation.execute())


class ToolCall:
    """Parsed model request to call a named tool with JSON-like arguments."""

    def __init__(self, name: str, args: dict, id: str | None = None):
        self.name = name
        self.args = copy.deepcopy(args)
        self.id = id or "call_" + uuid4().hex

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
        model, parameters, output = infer(func, param_descriptions)

        return Tool(
            name=func.__name__,
            declaration={
                "name": func.__name__,
                "description": description,
                "parameters": parameters,
            },
            function=func,
            default_params=default_params,
            add_to_registry=autobind,
            registry=registry,
            input_model=model,
            output_adapter=output,
        )

    return decorator
