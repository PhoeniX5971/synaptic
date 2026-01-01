import inspect
from typing import Callable, Any

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
    """
    Decorator to automatically create a Tool from a function.
    param_descriptions: optional dict mapping parameter names to descriptions
    """
    param_descriptions = param_descriptions or {}

    def decorator(func: Callable[..., Any]) -> Tool:
        sig = inspect.signature(func)
        properties = {}

        for name, param in sig.parameters.items():
            # SKIP 'self' or 'cls' so they don't end up in the AI's JSON schema
            # Happens in classes (where self is in the params of the method)
            if name in ["self", "cls"]:
                continue
            # Determine JSON type from annotation
            anno = param.annotation
            if anno in [int, float]:
                ptype = "number" if anno is float else "integer"
            elif anno is bool:
                ptype = "boolean"
            elif anno is str:
                ptype = "string"
            else:
                ptype = "string"  # fallback

            properties[name] = {
                "type": ptype,
                "description": param_descriptions.get(name, ""),  # default empty
            }

        declaration = {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(sig.parameters.keys()),
            },
        }

        tool = Tool(
            name=func.__name__,
            declaration=declaration,
            function=func,
            default_params=default_params,
            add_to_registry=autobind,
        )

        return tool

    return decorator
