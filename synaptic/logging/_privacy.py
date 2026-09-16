"""Redaction and finite JSON diagnostics before delivery to any sink."""

import json
import re
from urllib.parse import urlsplit, urlunsplit

PAYLOAD = {"prompt", "context", "response", "text", "args", "arguments", "result", "content"}
SECRET = re.compile(
    r"password|secret|(?:access|refresh|auth|bearer)[_-]?token|^token$|authorization|api.?key|cookie",
    re.I,
)


def sanitize(value, *, secrets=(), depth=0):
    if depth > 8:
        return "[depth limit]"
    if isinstance(value, dict):
        return {
            str(k): "[redacted]"
            if SECRET.search(str(k))
            else sanitize(v, secrets=secrets, depth=depth + 1)
            for k, v in list(value.items())[:100]
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(v, secrets=secrets, depth=depth + 1) for v in value[:100]]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
        value = re.sub(r"(?i)Bearer\s+[^\s]+|sk-[\w-]{6,}", "[redacted]", value)

        def endpoint(match):
            parts = urlsplit(match.group())
            host = (parts.hostname or "") + (f":{parts.port}" if parts.port else "")
            return urlunsplit((parts.scheme, host, parts.path, "", ""))

        value = re.sub(r"https?://[^\s]+", endpoint, value)
        return value[:2048]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return f"[{type(value).__name__}]"


def metadata_only(value):
    if isinstance(value, dict):
        return {key: metadata_only(item) for key, item in value.items() if key not in PAYLOAD}
    if isinstance(value, (list, tuple)):
        return [metadata_only(item) for item in value]
    return value


def prepare(event, payloads, secrets, redact, max_bytes):
    candidate = dict(event)
    if not payloads:
        candidate = metadata_only(candidate)
    candidate = sanitize(candidate, secrets=secrets)
    if redact:
        candidate = sanitize(redact(candidate), secrets=secrets)
    encoded = json.dumps(candidate, ensure_ascii=False)
    if len(encoded.encode()) > max_bytes:

        def bounded(value):
            return str(value).encode("utf-8")[:32].decode("utf-8", errors="ignore")

        return {
            "type": bounded(candidate.get("type", "event")),
            "truncated": True,
            **{
                key: bounded(candidate[key])
                for key in ("session_id", "run_id", "call_id")
                if key in candidate
            },
        }
    return candidate
