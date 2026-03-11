import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class NodeResult:
    """
    Return value from a Flow node.

    output: the value passed as state to the next node.
    next:   name of the next node to execute. None means stop the flow.
    """

    output: Any
    next: Optional[str] = None


class Flow:
    """
    Simple directed agent graph.

    Nodes are callables with signature: (state) -> NodeResult | any
    - If a node returns a NodeResult, flow routes to NodeResult.next.
    - If a node returns anything else (or NodeResult.next is None), flow stops.
    - Nodes can be sync or async — arun() handles both transparently.

    Usage:
        flow = Flow()

        @flow.node("fetch")
        def fetch(state):
            result = researcher.invoke(state)
            return NodeResult(output=result.message, next="summarise")

        @flow.node("summarise")
        def summarise(state):
            result = writer.invoke(state)
            return NodeResult(output=result.message, next=None)

        flow.entry("fetch")
        final = flow.run("What is the capital of France?")

    Conditional routing:
        @flow.node("router")
        def router(state):
            r = classifier.invoke(state)
            return NodeResult(output=state, next="agent_a" if "A" in r.message else "agent_b")
    """

    def __init__(self, max_steps: int = 50):
        self._nodes: Dict[str, Callable] = {}
        self._entry: Optional[str] = None
        self.max_steps = max_steps

    def node(self, name: str) -> Callable:
        """Decorator that registers a callable as a named node."""

        def decorator(fn: Callable) -> Callable:
            self._nodes[name] = fn
            return fn

        return decorator

    def add(self, name: str, fn: Callable) -> None:
        """Register a node without using the decorator syntax."""
        self._nodes[name] = fn

    def entry(self, name: str) -> None:
        """Set the entry-point node."""
        self._entry = name

    def run(self, state: Any) -> Any:
        """Run the flow synchronously. Raises RuntimeError if async nodes are used."""
        current = self._resolve_entry()

        for _ in range(self.max_steps):
            node = self._get_node(current)
            result = node(state)

            if inspect.iscoroutine(result):
                raise RuntimeError(
                    f"Node '{current}' is async — use arun() instead of run()."
                )

            if isinstance(result, NodeResult):
                state = result.output
                if result.next is None:
                    return state
                current = result.next
            else:
                return result

        raise RuntimeError(
            f"Flow exceeded max_steps={self.max_steps}. "
            "Increase max_steps or check for routing cycles."
        )

    async def arun(self, state: Any) -> Any:
        """Run the flow asynchronously. Works with both sync and async nodes."""
        current = self._resolve_entry()

        for _ in range(self.max_steps):
            node = self._get_node(current)
            result = node(state)

            if inspect.iscoroutine(result):
                result = await result

            if isinstance(result, NodeResult):
                state = result.output
                if result.next is None:
                    return state
                current = result.next
            else:
                return result

        raise RuntimeError(
            f"Flow exceeded max_steps={self.max_steps}. "
            "Increase max_steps or check for routing cycles."
        )

    def _resolve_entry(self) -> str:
        if self._entry is None:
            raise ValueError("No entry node set. Call flow.entry('node_name') first.")
        return self._entry

    def _get_node(self, name: str) -> Callable:
        if name not in self._nodes:
            raise ValueError(
                f"Node '{name}' not found. Registered nodes: {list(self._nodes)}"
            )
        return self._nodes[name]


class Pipeline:
    """
    Linear sequence of steps — each step receives the previous step's output.

    Simpler than Flow when there's no branching needed.
    Steps can be sync or async callables.

    Usage:
        pipeline = Pipeline([
            lambda s: researcher.invoke(s).message,
            lambda s: writer.invoke(s).message,
            lambda s: reviewer.invoke(s).message,
        ])

        final = pipeline.run("Write a report on climate change.")
        final = await pipeline.arun("Write a report on climate change.")
    """

    def __init__(self, steps: List[Callable]):
        self.steps = steps

    def run(self, state: Any) -> Any:
        for step in self.steps:
            result = step(state)
            if inspect.iscoroutine(result):
                raise RuntimeError(
                    "Async step detected — use arun() instead of run()."
                )
            state = result
        return state

    async def arun(self, state: Any) -> Any:
        for step in self.steps:
            result = step(state)
            if inspect.iscoroutine(result):
                result = await result
            state = result
        return state
