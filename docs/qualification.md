# Qualification — Control integration follow-up (T-011)

Measured 2026-09-18 against the integration follow-up to clean baseline
`3eb9956d9d9441dedf977c2a6ee76696e1ab42c9`. S-008 preserves the prior 302-test/90-file
qualification. This review is mapped in integration-followup.md; older findings and
commands remain in finding-report.md, readiness-followup.md and progress-log.md.

## Measured compatibility

| Component | Version / result |
| --- | --- |
| Runtime | CPython 3.13.8, Linux x86_64; 360 tests passed in a fresh uv.lock environment |
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

- Historical gates: 176 original baseline tests, 238 tests in S-007, 302 tests in S-008.
- Four public Model/Retry/budget/reconciliation integration cases failed before this fix.
  The final suite adds 58 public integration cases and preserves all earlier tests.
- Control's two execution failures were reproduced in five cases before the fix:
  `pytest tests/test_execution_recovery.py -q --tb=short` → **five failures**.
- Current final full gate: **360 passed, four warnings**: two expected history
  truncations and two upstream Vertex/protobuf deprecations. RuntimeWarning is an error.
- This includes the original lifecycle/ownership/provider suites and three real MCP
  transport tests, plus sync/async/stream/external executor receipt and cancellation
  coverage, observers, output validation, scheduler queues and serialized reconciliation.
  New tests combine real public Model/Agent/Retry APIs with in-process adapter fixtures,
  avoiding live services and internal-runner-only evidence.

Reproduce with uv 0.12.10 and CPython 3.13.8 from the repository root:

```sh
UV_CACHE_DIR=/tmp/synaptic-uv-cache UV_PROJECT_ENVIRONMENT=/tmp/synaptic-integration-env uv sync --frozen --all-extras --python 3.13.8
/tmp/synaptic-integration-env/bin/python scripts/check_quality.py
```

The gate prints measured interpreter/tool versions, runs fatal Ruff, checks **all 92
runtime-package source files** with mypy and then runs the full test suite. Mypy uses
`--config-file pyproject.toml --no-incremental synaptic`, normal import following and
`check_untyped_defs=true`; no project module is excluded or followed with skip. Tests
and scripts receive fatal lint and execution checks; they are not claimed as strictly
type-annotated. T-010's historic pre-fix full-body analysis found **132 errors in 28 of 88 files**;
current analysis reports **zero errors in 92 files**. The older 81-error/ten-module
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

Reproduce with the same fresh development interpreter:

```sh
/tmp/synaptic-integration-env/bin/python scripts/check_release.py
```

This passed against the final runtime: wheel and sdist both include `synaptic/py.typed`;
all ten isolated install cases passed. Core, OpenAI and all-extras targets also ran strict
mypy from outside the checkout using the target Python for package discovery. Positive
assert_type examples passed for Model/Agent sync/async/streaming, proposal return types,
history and core records. **All 12 intentionally invalid API lines were rejected in
each of the three targets**. Consumer config contains no ignores, Any casts or optional
SDK workarounds. Core-only targets contain no provider SDKs or development dependencies;
the checker is supplied by the qualified development host. The first harness run expected
absolute filenames; mypy's relative diagnostics were correctly recognized on rerun.

The credential-free offline example ran with isolated Python (`-I`) against those three
installed wheels. `uv pip check` passed for every install. The release script builds in
a temporary directory, verifies artifact contents, exports the frozen constraints and
never publishes. Dependencies/build tools may be downloaded; no provider requests or
credentials are used. No unpinned-consumer equivalence is claimed.

## Tracked automation and release prerequisites

`.github/workflows/qualification.yml` runs the frozen quality and release scripts with
Ubuntu 24.04, CPython 3.13.8 and uv 0.12.10. Action tags were resolved against official
repositories and pinned to immutable commits. Permissions are read-only and there is no
publish step. The scripts passed locally; **no remote CI run is claimed**. The configured
Ubuntu runner image has not been separately qualified by a remote execution.

CONTRIBUTING.md, SECURITY.md, CHANGELOG.md and docs/releasing.md link the existing records.
A private vulnerability reporting route has not been designated/verified by the maintainer;
SECURITY.md records that prerequisite without inventing a contact. Only the maintainer
chooses versions, tags, reporting settings and publication.

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
Recovery and failed-attempt archives are independently retained and are not automatically
pruned by context budgets. The budget bounds the supplied history estimator, not total
request tokens or archive memory. The execution extension is in-memory evidence, not
crash-proof durable exactly-once storage.
Serialization requires JSON-compatible results and application-supplied token counters.
Diagnostics can report/drop events and are not a reliable journal. These boundaries are
specified in contracts.md, not hidden as future fixes.
