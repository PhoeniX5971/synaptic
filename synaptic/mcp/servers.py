from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StdioMCPServer:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class HttpMCPServer:
    name: str
    url: str
    headers: Optional[Dict[str, str]] = None


MCPServer = StdioMCPServer | HttpMCPServer
