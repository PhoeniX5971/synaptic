from typing import Callable, Any


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
    ) -> None:
        self.name = name
        self.declaration = declaration
        if not callable(function):
            raise ValueError("Function must be a callable")
        else:
            self.function = function
        self.default_params = default_params or {}

    def run(self, **kwargs) -> Any:
        # Merge default params with runtime kwargs
        final_kwargs = {**self.default_params, **kwargs}
        return self.function(**final_kwargs)


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
