from typing import Callable, Any

class Tool:
    def __init__(
        self,
        name: str,
        declaration: dict,
        function: Callable[..., Any] = lambda : None,
        default_params: dict | None = None
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
