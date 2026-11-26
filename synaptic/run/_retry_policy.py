"""Transient classification and finite retry timing independent of tool effects."""

import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


# Matched by class name anywhere in the MRO, so no SDK or HTTP library is imported here.
# TransportError is the httpx base for connections and streams that break or time out;
# SDKs let it through unwrapped once a stream is being read.
_TRANSIENT_TYPES = {"APIConnectionError", "APITimeoutError", "RateLimitError", "TransportError"}


def transient(error):
    """Whether an error is a temporary connection, timeout, rate or server failure.

    A status-less SDK `APIError` (an error event sent inside a stream) is not treated
    as temporary: it can also report a permanent server-side failure.
    """
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return (
        isinstance(error, (ConnectionError, TimeoutError))
        or status in (408, 429)
        or isinstance(status, int)
        and 500 <= status < 600
        or bool(_TRANSIENT_TYPES & {cls.__name__ for cls in type(error).__mro__})
    )


def retry_after(error):
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return 0.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            return max(
                0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            )
        except (TypeError, ValueError):
            return 0.0


class RetryPolicy:
    def __init__(
        self,
        retries=3,
        delay=0.5,
        *,
        multiplier=2.0,
        max_delay=30.0,
        jitter=0.2,
        deadline=60.0,
        exceptions=None,
        predicate=None,
    ):
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if any(not math.isfinite(v) for v in (delay, multiplier, max_delay, jitter, deadline)):
            raise ValueError("Retry timings must be finite")
        if delay < 0 or multiplier < 1 or max_delay < 0 or not 0 <= jitter <= 1 or deadline <= 0:
            raise ValueError("Invalid retry timing configuration")
        self.retries, self.delay = retries, delay
        self.multiplier, self.max_delay, self.jitter, self.deadline = (
            multiplier,
            max_delay,
            jitter,
            deadline,
        )
        self.predicate = (
            predicate or (lambda error: isinstance(error, exceptions))
            if exceptions
            else predicate or transient
        )

    def pause(self, attempt, error, remaining):
        if attempt >= self.retries or not self.predicate(error):
            return None
        delay = min(self.max_delay, self.delay * self.multiplier ** min(attempt, 32))
        delay *= 1 - self.jitter * random.random()
        required = retry_after(error)
        if required > self.max_delay:
            return None  # Never retry earlier than the server asks.
        delay = max(delay, required)
        return delay if delay < remaining else None
