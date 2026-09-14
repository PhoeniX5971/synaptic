import json
from pathlib import Path
from typing import Any, Dict


class JSONLWriter:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def write(self, event: Dict[str, Any]) -> None:
        self._f.write(json.dumps(event, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
