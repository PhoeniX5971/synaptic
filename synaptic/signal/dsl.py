import json
import re
from typing import Callable, Dict, List, Optional

from ..core.base.base_model import ToolCallArgsDelta
from ..core.tool import ToolCall
from .events import (
    BlockDelta, BlockDone, BlockStarted,
    SignalEvent, TextDelta, ToolCallDone, ToolCallStarted,
)

_DSL_FRAGMENT = """
## Output Protocol
Always wrap your output in blocks. Never output bare text.
- Plain responses: <message>your text here</message>
- Tool calls: <tool name="TOOL_NAME">{"arg": "value"}</tool>
"""

_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def inject_instructions(model, tools) -> None:
    lines = [
        f"- {t.declaration['name']}({', '.join(t.declaration.get('parameters', {}).get('properties', {}))})"
        f": {t.declaration.get('description', '')}"
        for t in tools
    ]
    suffix = "\nAvailable tools:\n" + "\n".join(lines) if lines else ""
    model.instructions = (model.instructions or "") + _DSL_FRAGMENT + suffix


def _tool_handler(evt: SignalEvent) -> List[SignalEvent]:
    if isinstance(evt, BlockStarted):
        return [ToolCallStarted(name=evt.attrs.get("name", ""))]
    if isinstance(evt, BlockDelta):
        return [ToolCallArgsDelta(evt.attrs.get("name", ""), evt.delta, evt.snapshot)]
    if isinstance(evt, BlockDone):
        try:
            args = json.loads(evt.content.strip())
        except (json.JSONDecodeError, ValueError):
            args = {}
        return [ToolCallDone(call=ToolCall(name=evt.attrs.get("name", ""), args=args))]
    return []


def _message_handler(evt: SignalEvent) -> List[SignalEvent]:
    if isinstance(evt, BlockDelta):
        return [TextDelta(text=evt.delta)]
    return []


class DSLParser:
    def __init__(self) -> None:
        self._handlers: Dict[str, Optional[Callable]] = {
            "tool": _tool_handler,
            "message": _message_handler,
        }
        self.reset()

    def reset(self) -> None:
        self._buf = ""
        self._state = "normal"
        self._block_name = ""
        self._attrs: Dict[str, str] = {}
        self._snapshot = ""
        self._close = ""

    def register(self, name: str, handler: Optional[Callable] = None) -> None:
        self._handlers[name] = handler

    def _handle(self, evt: SignalEvent) -> List[SignalEvent]:
        h = self._handlers.get(getattr(evt, "type", ""))
        return h(evt) if h else []

    def push(self, delta: str) -> List[SignalEvent]:
        events: List[SignalEvent] = []
        self._buf += delta

        while True:
            if self._state == "normal":
                idx = self._buf.find("<")
                if idx == -1:
                    self._buf = self._buf[-1:] if self._buf else ""
                    break
                self._buf = self._buf[idx + 1:]
                self._state = "tag_open"

            elif self._state == "tag_open":
                idx = self._buf.find(">")
                if idx == -1:
                    break
                tag_str = self._buf[:idx]
                self._buf = self._buf[idx + 1:]
                parts = tag_str.split(None, 1)
                self._block_name = parts[0]
                self._attrs = dict(_ATTR_RE.findall(parts[1] if len(parts) > 1 else ""))
                self._close = f"</{self._block_name}>"
                self._snapshot = ""
                self._state = "in_block"
                evt = BlockStarted(type=self._block_name, attrs=self._attrs)
                events.append(evt)
                events.extend(self._handle(evt))

            elif self._state == "in_block":
                idx = self._buf.find(self._close)
                if idx == -1:
                    safe = max(0, len(self._buf) - len(self._close) + 1)
                    if safe:
                        emit = self._buf[:safe]
                        self._snapshot += emit
                        evt = BlockDelta(type=self._block_name, delta=emit, snapshot=self._snapshot, attrs=self._attrs)
                        events.append(evt)
                        events.extend(self._handle(evt))
                        self._buf = self._buf[safe:]
                    break
                chunk = self._buf[:idx]
                if chunk:
                    self._snapshot += chunk
                    evt = BlockDelta(type=self._block_name, delta=chunk, snapshot=self._snapshot, attrs=self._attrs)
                    events.append(evt)
                    events.extend(self._handle(evt))
                evt = BlockDone(type=self._block_name, content=self._snapshot, attrs=self._attrs)
                events.append(evt)
                events.extend(self._handle(evt))
                self._buf = self._buf[idx + len(self._close):]
                self._block_name = self._snapshot = self._close = ""
                self._attrs = {}
                self._state = "normal"

        return events

    def flush(self) -> List[SignalEvent]:
        return []
