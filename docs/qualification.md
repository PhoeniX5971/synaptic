# Qualification — completed hardening follow-up

Measured 2026-09-16 against the implementation following baseline
6a8b98701b0f3cba95ac2ea7e16d4a1d44fe7b0f. Historical first-tranche commands/results
remain in progress-log.md (S-001–S-003). Final findings are mapped in finding-report.md.

## Measured compatibility

| Component | Version / result |
| --- | --- |
| Runtime | CPython 3.13.8, Linux; 238 tests passed in a fresh uv.lock environment |
| MCP SDK / protocol | 1.30.0 / 2025-11-25; real subprocess stdio and loopback Streamable HTTP |
| HTTP foundations | httpx 0.28.1, anyio 4.12.1, uvicorn 0.52.4, Starlette 1.6.0 |
| Schema | Pydantic 2.12.5, jsonschema 4.26.0 |
| OpenAI-compatible | OpenAI 2.16.0; real SDK MockTransport parsing/configuration |
| Anthropic | 0.84.0; native typed message/event fixtures |
| Gemini | google-genai 1.61.0; typed response/configuration fixtures |
| Vertex | google-cloud-aiplatform 1.71.1; native async/response/configuration fixtures |
| Together | 2.2.0; bounded sync iterator bridge/Chat response fixtures |
| xAI | 1.8.1; native protobuf response wrapper/configuration/history fixtures |
| Checks | pytest 9.0.2, Ruff 0.15.6, mypy 1.19.1 |

SDK fixtures do not qualify live provider models or every option. No credentials or
provider network requests were used. Library dependency ranges and Python >=3.11 are
compatibility declarations; only the versions/platform above are measured. MCP 2.x
is excluded, and other 1.x/protocol paths/extensions are not implied by the range.

Reviewed primary references and installed signatures:
[MCP ClientSession 1.30.0](https://github.com/modelcontextprotocol/python-sdk/blob/v1.30.0/src/mcp/client/session.py),
[MCP HTTP 1.30.0](https://github.com/modelcontextprotocol/python-sdk/blob/v1.30.0/src/mcp/client/streamable_http.py),
[OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat),
[Google Gen AI SDK](https://googleapis.github.io/python-genai/),
[xAI chat API](https://docs.x.ai/developers/grpc-api-reference/chat).

## Regression evidence

- Reviewed baseline: 176 passed, two expected history warnings.
- Four follow-up defects reproduced in six cases before implementation:
  `pytest tests/test_followup.py -q` → six failures.
- Intermediate in-process suite: 190 passed. Further boundary/provider cases were added.
- Fresh locked environment:
  `UV_CACHE_DIR=/tmp/synaptic-uv-cache UV_PROJECT_ENVIRONMENT=/tmp/synaptic-qualification-env uv sync --frozen --all-extras --python .venv/bin/python`.
- `/tmp/synaptic-qualification-env/bin/pytest tests -q -W error::RuntimeWarning --tb=short`
  → **238 passed, four warnings**: two expected history truncations and two upstream
  Vertex/protobuf deprecations. All original tests remain; adapter ID assertions were
  updated to the new native/stable ID contract rather than fixed synthesized indices.
- The suite includes three real local MCP transport tests. Thread/SDK mock-HTTP tests
  also require normal loop wakeup permissions. A sandbox thread-capacity test timed out,
  then passed unchanged with local permissions; the sandbox mock-HTTP run was interrupted
  after it stalled and passed with those permissions. No production workaround was added.

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

## Static checks and implementation limits

- `.venv/bin/ruff check --target-version py311 --select E9,F63,F7,F82 synaptic tests scripts` passed.
- `.venv/bin/mypy --follow-imports=skip --check-untyped-defs synaptic/core/_awaitables.py synaptic/core/_completion.py synaptic/core/_metadata.py synaptic/core/capabilities.py synaptic/mcp/_connection.py synaptic/mcp/_discovery.py synaptic/mcp/_transport.py synaptic/mcp/results.py synaptic/mcp/errors.py synaptic/mcp/session.py` passed: ten focused modules.
- `.venv/bin/mypy --follow-imports=skip synaptic` reports **81 errors in 14 files**,
  checking 88 source files. Dynamic mixins, optional annotations and provider SDK typing
  remain incomplete; there is no whole-project type-clean claim or attribution of every
  issue to the old baseline.
- `git diff --check` passed. Changed/new source modules are at most 200 lines. Formatting
  was scoped to changed implementation; nine unrelated formatting-only diffs were backed
  up, inspected and proven syntax-tree equivalent before restoration. Automatic review
  rejected an earlier broad restoration proposal; it did not execute.

Discovery byte limits apply after SDK decoding. Idle eviction occurs on next use.
Cancelling a thread await does not stop the effect or free actual capacity; sync deadlines
cannot preempt blocking functions. Serialization requires JSON-compatible results and
application-supplied token counters. Diagnostics can report/drop events and are not a
reliable journal. These boundaries are specified in contracts.md, not hidden as future fixes.
