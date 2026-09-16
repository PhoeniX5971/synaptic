# Retry policy and rationale

Retry wraps safe model generation. Autorun effects prohibit automatic retries; unknown
producers need retry_safe=True. Any yielded stream chunk is a replay barrier. A tool
failure is not a signal to repeat uncertain effects. Generic @retry remains available
for callers that explicitly own repetition safety.

`_retry_policy.py` classifies transport/timeouts, 408/429/5xx and known SDK transient
exceptions. Authentication, validation and programming errors do not match the default.
It validates finite timing, nonnegative retry count/delay, bounded exponential backoff
and jitter. Retry-after is a minimum: if it exceeds the maximum delay or remaining
deadline, do not retry early. Exceptions propagate with their original context.

Async calls/streams have a total deadline; cancelling during backoff propagates.
Sync deadlines bound retry scheduling but cannot preempt an already-running function.
Sources close in streaming finalization. The decorator's default broad filter retains
legacy caller responsibility; the model wrapper uses transient classification.

Validation: test_retry_contract, test_retry_boundaries and test_run pass 41 tests.
They cover timing/status classification, server delay, deadlines, zero/invalid counts,
pre-output retry, visible-output failure and the uncertain-effect barrier.
