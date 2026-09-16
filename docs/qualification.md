# Qualification — Control readiness follow-up (T-010)

Measured 2026-09-16 after reviewing Control against Synaptic baseline
`cac43ac2c5184926b4e1b89c924e76346e961969`. This record supersedes S-007's typing
limitation. Earlier commands/results remain in progress-log.md; the original findings
remain in finding-report.md and the new review is mapped in readiness-followup.md.

## Measured compatibility

| Component | Version / result |
| --- | --- |
| Runtime | CPython 3.13.8, Linux x86_64; 302 tests passed in a fresh uv.lock environment |
| MCP SDK / protocol | 1.30.0 / 2025-11-25; real subprocess stdio and loopback Streamable HTTP |
| HTTP foundations | httpx 0.28.1, anyio 4.12.1, uvicorn 0.52.4, Starlette 1.6.0 |
| Schema | Pydantic 2.12.5, jsonschema 4.26.0 |
| OpenAI-compatible | OpenAI 2.16.0; real SDK MockTransport parsing/configuration |
| Anthropic | 0.84.0; native typed message/event fixtures |
| Gemini | google-genai 1.61.0; typed response/configuration fixtures |
| Vertex | google-cloud-aiplatform 1.71.1; native async/response/configuration fixtures |
| Together | 2.2.0; bounded sync iterator bridge/Chat response fixtures |
| xAI | 1.8.1; native protobuf response wrapper/configuration/history fixtures |
| Checks | pytest 9.0.2, Ruff 0.15.6, mypy 1.19.1, types-jsonschema 4.26.0.20260518 |

SDK fixtures do not qualify live provider models or every option. No credentials or
provider network requests were used. Library dependency ranges and Python >=3.11 are
a language floor and dependency declarations; only the versions/platform above are measured.
Windows, macOS, Python 3.11/3.12/3.14 and unpinned SDK resolutions are unqualified.
Before broadening support, rerun the full gate, real MCP transports, cancellation/process
cleanup and isolated installs on the proposed platform/version. The current install
harness uses POSIX venv paths; it is not a Windows qualification harness. No remote CI
execution or live-service qualification is claimed. MCP 2.x
is excluded, and other 1.x/protocol paths/extensions are not implied by the range.

Reviewed primary references and installed signatures:
[MCP ClientSession 1.30.0](https://github.com/modelcontextprotocol/python-sdk/blob/v1.30.0/src/mcp/client/session.py),
[MCP HTTP 1.30.0](https://github.com/modelcontextprotocol/python-sdk/blob/v1.30.0/src/mcp/client/streamable_http.py),
[OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat),
[Google Gen AI SDK](https://googleapis.github.io/python-genai/),
[xAI chat API](https://docs.x.ai/developers/grpc-api-reference/chat).

## Regression evidence and reproducible full gate

- Historical gates: 176 original baseline tests, then 238 tests in S-007.
- Control's two execution failures were reproduced in five cases before the fix:
  `pytest tests/test_execution_recovery.py -q --tb=short` → **five failures**.
- Current final full gate: **302 passed, four warnings**: two expected history
  truncations and two upstream Vertex/protobuf deprecations. RuntimeWarning is an error.
- This includes the original lifecycle/ownership/provider suites and three real MCP
  transport tests, plus sync/async/stream/external executor receipt and cancellation
  coverage, observers, output validation, scheduler queues and serialized reconciliation.

Reproduce with uv 0.12.10 and CPython 3.13.8 from the repository root:

```sh
UV_CACHE_DIR=/tmp/synaptic-uv-cache UV_PROJECT_ENVIRONMENT=/tmp/synaptic-readiness-env uv sync --frozen --all-extras --python 3.13.8
/tmp/synaptic-readiness-env/bin/python scripts/check_quality.py
```

The gate prints measured interpreter/tool versions, runs fatal Ruff, checks **all 90
runtime-package source files** with mypy and then runs the full test suite. Mypy uses
`--config-file pyproject.toml --no-incremental synaptic`, normal import following and
`check_untyped_defs=true`; no project module is excluded or followed with skip. Tests
and scripts receive fatal lint and execution checks; they are not claimed as strictly
type-annotated. Fresh pre-fix full-body analysis found **132 errors in 28 of 88 files**;
current analysis reports **zero errors in 90 files**. The older 81-error/ten-module
reports used a different gate and are preserved only as historical measurements.

Local stdio/socket/thread wakeup fixtures need ordinary local permissions. Earlier
sandbox stalls passed unchanged with those permissions; no production workaround or
transport skip was added. Provider fixtures use SDK objects/MockTransport and never
contact provider services. The final gate passed in a newly created frozen environment.

## Packaging evidence

Built wheel and sdist, then checked all ten isolated cases: core, mcp, openai, deepseek,
claude, gemini, vertex, together, xai and all. Every case passed core lazy-import checks,
selected provider imports and `uv pip check`. Core had no OpenAI distribution, and
requesting its adapter returned the actionable missing-extra error. The final wheel was
checked independently of the source checkout using isolated Python (`-I`).

Reproduce from a checkout with uv 0.12.10:

```sh
uv lock --check
uv build --out-dir /tmp/synaptic-dist
uv export --frozen --all-extras --no-emit-project --format requirements-txt --output-file /tmp/synaptic-constraints.txt
python scripts/check_installs.py /tmp/synaptic-dist/synaptic-0.1.0-py3-none-any.whl --constraints /tmp/synaptic-constraints.txt
```

Dependency downloads required network permission after sandbox DNS failures. An initial
final-wheel install attempt failed because its old temporary constraint export no longer
existed; a fresh export was generated and every final-wheel case passed. Build output
and temporary environments live outside the repository; no artifacts/credentials are committed.
`uv.lock` pins the complete development resolution; no claim is made that an unpinned
consumer resolves identical transitive dependencies.

## Typing boundaries and implementation limits

- `scripts/check_quality.py` passed: fatal Ruff, full-package mypy, complete tests.
- JSON Schema uses pinned `types-jsonschema==4.26.0.20260518` in the development group.
- Only `vertexai`, `vertexai.*`, `xai_sdk` and `xai_sdk.*` have
  `ignore_missing_imports=true`: the pinned SDK distributions omit `py.typed` metadata.
  Their imported SDK symbols are untyped at that boundary; all Synaptic wrappers and
  mixin bodies are still checked. SDK response/configuration fixtures exercise the
  boundary at runtime. Remove those overrides when upstream ships usable typing.
- There are no project-wide ignores, excluded owned files or `follow_imports=skip`.
  This is a full-package baseline typing gate, not strict typing of every dynamic SDK
  payload or complete annotations for every public argument.
- `git diff --check`, fatal lint and changed/new source module <=200-line checks passed.
  Formatting was scoped to touched implementation/tests.

Discovery byte limits apply after SDK decoding. Idle eviction occurs on next use.
Cancelling an already-running thread or opaque external executor cannot stop/prove absence
of effects; it remains uncertain. Not-yet-admitted cooperative work rechecks cancellation.
Earlier receipts survive both failure and cancellation; reconciliation cannot replace them.
The execution extension is in-memory evidence, not crash-proof durable exactly-once storage.
Serialization requires JSON-compatible results and application-supplied token counters.
Diagnostics can report/drop events and are not a reliable journal. These boundaries are
specified in contracts.md, not hidden as future fixes.
