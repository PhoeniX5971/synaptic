import inspect
from typing import Callable, Any
import types

TOOL_REGISTRY = {}
_registry_callbacks = []  # FOR SUBSCRIBED MODELS


def register_callback(fn: Callable[[], None]):
    """
    SUBSCRIBES A MODEL TO TOOL REGISTRY CHANGES
    Takes a function with no arguments from the model
    on each and adds it to the registry callbacks list
    this list is called on each change to the tool registry
    """
    _registry_callbacks.append(fn)


def _notify_change():
    """
    Calls all subscribed model callbacks to notify them of a change in the tool registry
    """
    for cb in _registry_callbacks:
        cb()


class Tool:
    """
    Tool type for better support of tools, allows for easier access of it's attributes.
    Constructor Attributes:
    """

    def __init__(
        self,
        name: str,
        declaration: dict,
        function: Callable[..., Any] = lambda: None,
        default_params: dict | None = None,
        add_to_registry: bool = True,
    ) -> None:
        self.name = name
        self.declaration = declaration
        if not callable(function):
            raise ValueError("Function must be a callable")
        else:
            self.function = function
            self.is_async = inspect.iscoroutinefunction(function)

        self.default_params = default_params or {}
        if add_to_registry:
            TOOL_REGISTRY[name] = self
            _notify_change()

        if self.is_async:
            self.run = self._run_async
        else:
            self.run = self._run_sync

        def __get__(self, instance, owner):
            if instance is None:
                return self

            bound_fn = types.MethodType(self.function, instance)

            tool = Tool(
                name=self.name,
                declaration=self.declaration,
                function=bound_fn,
                default_params=self.default_params,
                add_to_registry=False,
            )

            return tool

    def _run_sync(self, **kwargs):
        final = {**self.default_params, **kwargs}
        return self.function(**final)

    async def _run_async(self, **kwargs):
        final = {**self.default_params, **kwargs}
        return await self.function(**final)

    def __repr__(self) -> str:
        decl_summary = {
            k: self.declaration[k]
            for k in ("name", "description")
            if k in self.declaration
        }
        return f"<Tool name={self.name!r}, declaration={decl_summary}, default_params={self.default_params}>"


class ToolCall:
    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args

    def get_arg(self, key: str):
        return self.args.get(key)

    def list_args(self):
        return self.args.keys()

    def __repr__(self):
        args_repr = {k: v for k, v in self.args.items() if len(str(v)) < 50}
        args_str = ", ".join(f"{k}={v!r}" for k, v in args_repr.items())
        return f"<ToolCall(name={self.name!r}, args={{ {args_str} }})>"


def autotool(
    description: str = "",
    param_descriptions: dict[str, str] | None = None,
    default_params: dict | None = None,
    autobind: bool = True,
):
    param_descriptions = param_descriptions or {}

    def decorator(func: Callable[..., Any]):
        sig = inspect.signature(func)
        properties: dict[str, dict] = {}
        required_params: list[str] = []

        for name, param in sig.parameters.items():
            # skip instance/class receivers
            if name in ("self", "cls"):
                continue

            # determine JSON type (simplified)
            anno = param.annotation
            if anno in [int, float]:
                ptype = "number" if anno is float else "integer"
            elif anno is bool:
                ptype = "boolean"
            elif anno is str:
                ptype = "string"
            else:
                ptype = "string"

            properties[name] = {
                "type": ptype,
                "description": param_descriptions.get(name, ""),
            }

            if param.default is inspect.Parameter.empty:
                required_params.append(name)

        parameters = {"type": "object", "properties": properties}
        if required_params:
            parameters["required"] = required_params

        declaration = {
            "name": func.__name__,
            "description": description,
            "parameters": parameters,
        }

        return Tool(
            name=func.__name__,
            declaration=declaration,
            function=func,
            default_params=default_params,
            add_to_registry=autobind,
        )

    return decorator
