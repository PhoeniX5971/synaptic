import json
from typing import List

from ..core.base.base_model import ToolCallArgsDelta
from ..core.tool import ToolCall
from .events import SignalEvent, TextDelta, ToolCallDone, ToolCallStarted

_OPEN = '<tool name="'
_OPEN_END = '">'
_CLOSE = "</tool>"

_DSL_FRAGMENT = """
## Streaming Tool Protocol
When calling tools, use this streaming text format to enable live previews:
<tool name="TOOL_NAME">
{"arg": "value"}
</tool>
Replace TOOL_NAME with the tool name and provide a JSON object as arguments.
Prefer this format over native function calls when streaming responses.
"""


def inject_instructions(model, tools) -> None:
    """Append DSL tool-call instructions + tool schemas to model.instructions."""
    lines = [
        f"- {t.declaration['name']}({', '.join(t.declaration.get('parameters', {}).get('properties', {}))})"
        f": {t.declaration.get('description', '')}"
        for t in tools
    ]
    model.instructions = (model.instructions or "") + _DSL_FRAGMENT + "\nAvailable tools:\n" + "\n".join(lines)


class DSLParser:
    """Incremental state-machine parser for the synaptic text DSL.

    Feed text deltas via push(); each call returns any SignalEvents produced.
    Call flush() at end of stream to drain buffered plain text.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._state = "normal"   # normal | tag_open | in_args
        self._tool_name = ""
        self._snapshot = ""

    def push(self, delta: str) -> List[SignalEvent]:
        events: List[SignalEvent] = []
        self._buf += delta

        while True:
            if self._state == "normal":
                idx = self._buf.find(_OPEN)
                if idx == -1:
                    safe = max(0, len(self._buf) - len(_OPEN) + 1)
                    if safe:
                        events.append(TextDelta(text=self._buf[:safe]))
                        self._buf = self._buf[safe:]
                    break
                if idx:
                    events.append(TextDelta(text=self._buf[:idx]))
                self._buf = self._buf[idx + len(_OPEN):]
                self._state = "tag_open"

            elif self._state == "tag_open":
                idx = self._buf.find(_OPEN_END)
                if idx == -1:
                    break
                self._tool_name = self._buf[:idx]
                self._buf = self._buf[idx + len(_OPEN_END):]
                self._snapshot = ""
                self._state = "in_args"
                events.append(ToolCallStarted(name=self._tool_name))

            elif self._state == "in_args":
                idx = self._buf.find(_CLOSE)
                if idx == -1:
                    if self._buf:
                        self._snapshot += self._buf
                        events.append(ToolCallArgsDelta(self._tool_name, self._buf, self._snapshot))
                        self._buf = ""
                    break
                chunk = self._buf[:idx]
                if chunk:
                    self._snapshot += chunk
                    events.append(ToolCallArgsDelta(self._tool_name, chunk, self._snapshot))
                try:
                    args = json.loads(self._snapshot.strip())
                except (json.JSONDecodeError, ValueError):
                    args = {}
                events.append(ToolCallDone(call=ToolCall(name=self._tool_name, args=args)))
                self._buf = self._buf[idx + len(_CLOSE):]
                self._tool_name = self._snapshot = ""
                self._state = "normal"

        return events

    def flush(self) -> List[SignalEvent]:
        events: List[SignalEvent] = []
        if self._buf and self._state == "normal":
            events.append(TextDelta(text=self._buf))
            self._buf = ""
        return events
