from synaptic.core import autotool


@autotool(
    description="A tool that adds two numbers",
    param_descriptions={"a": "First integer", "b": "Second integer"},
)
def add_numbers(a: int, b: int) -> int:
    return a + b
