import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JSONLWriter:
    def __init__(self, path, *, max_bytes=1024 * 1024, backups=3):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )

    def write(self, event):
        # Call handleError as a raised failure so the diagnostic bus can report it.
        line = json.dumps(event, ensure_ascii=False)
        record = logging.LogRecord("synaptic", logging.INFO, "", 0, line, (), None)
        self.handler.acquire()
        try:
            if self.handler.shouldRollover(record):
                self.handler.doRollover()
            self.handler.stream.write(line + "\n")
            self.handler.flush()
        finally:
            self.handler.release()

    def close(self):
        self.handler.close()
