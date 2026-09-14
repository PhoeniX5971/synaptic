Improve Synaptic into a reliable, reusable model and agent framework.

Fix the defects below and implement the improvements where they provide concrete value. API changes, refactoring, new features, and replacement of weak implementations are allowed. Preserve useful functionality and document migrations for breaking changes.

Work in ~/code/ai/framework. Keep Synaptic general-purpose: desktop automation, Control plugin packaging, application-specific grants, and PC resource scheduling belong to consumers such as Control.

**Baseline and working rules**

The reviewed baseline was commit:
1f9a963c9fe68c6951410072eb6469c1572ce59b

The working tree contained modifications to:
- PLAN.MD
- synaptic/__init__.py
- synaptic/agent/agent.py
- synaptic/core/_runner.py
- synaptic/mcp/session.py

It also contained untracked permission/logging modules and project records.

Inspect the current state before changing anything. Preserve existing work and distinguish committed behavior from working-tree changes. Reproduce findings against the actual current implementation; do not assume this review remains current.

Use the production-coding-workflow and coding-agent-handoff-protocol skills. Follow repository instructions, maintain architecture/rationale/issue/progress records, and make cohesive commits with relevant validation.

Use the configured author and committer:
PhoeniX5971 <moh845195@gmail.com>

Do not publish, change remotes, rewrite history, or modify sibling projects.

**Reproduced defects**

1. MCP normalization discards information

Location:
synaptic/mcp/results.py::normalize_result

Observed:
When structured content exists, normalization returns it immediately and discards accompanying image/text content and metadata. Other paths flatten text or reduce tool errors to a small dictionary.

Required:
- Preserve structured values, typed content blocks, error indicators and metadata together.
- Preserve image/audio/resource modalities and their provenance.
- Distinguish protocol errors, transport failures and tool-reported failures.
- Make text extraction an explicit convenience operation.
- Provide a documented compatibility path for callers expecting the previous simplified result.
- Avoid leaking arbitrary SDK implementation objects through the public normalized contract.

Acceptance:
Mixed text/image/structured output survives normalization. Error details remain distinguishable. Supported structured values round-trip correctly.

2. MCP discovery ignores pagination

Location:
synaptic/mcp/session.py::list_tools

Observed:
Only the first response's tools are returned; the continuation cursor is ignored.

Required:
- Consume every supported discovery page.
- Bound page count, result count, payload size and total time.
- Detect repeated cursors.
- Define partial-failure behavior explicitly.
- Publish a refreshed catalog atomically after successful validation.

Acceptance:
Multiple pages are discovered. Repeated cursors terminate safely. Failed refresh does not silently replace a valid catalog with an incomplete one.

3. Failed MCP sessions can be treated as connected

Location:
synaptic/mcp/session.py::connect

Observed:
The existing-task branch waits for readiness and returns without checking the stored error or whether a live session exists.

Required:
- Implement explicit connection lifecycle states.
- Coalesce concurrent connection attempts.
- Propagate startup failures to every waiter.
- Handle failures before transport setup, including imports/configuration.
- Detect a terminated owner task and invalidate stale session state.
- Bound connection, shutdown and reconnection.
- Ensure cancellation during startup releases waiters.
- Preserve the useful invariant that transport context managers enter and exit in their owning task.

Acceptance:
Initial failure, simultaneous callers, post-connect transport death, cancellation during initialization, repeated close and reconnect all terminate predictably without stale sessions or hanging waiters.

4. Async invocation loses token usage

Location:
synaptic/core/_runner.py::ainvoke

Observed:
A fake provider emitted final usage of 12 input tokens and 3 output tokens. The returned ResponseMem reported 0 and 0.

Required:
- Preserve usage from final provider events.
- Apply equivalent accounting to streamed history and agent-generated response records.
- Distinguish unavailable usage from measured zero.
- Preserve relevant completion metadata consistently across invocation modes.

Acceptance:
invoke, ainvoke and astream/history report equivalent usage for equivalent provider fixtures.

5. Stream retries duplicate previously emitted output

Location:
synaptic/run/retry.py::Retry.astream

Observed:
A stream yielding "first", then failing, produced:
["first", "first", "last"]
after retry.

Its documentation incorrectly implies previously emitted chunks will not be repeated.

Required:
- Define retry semantics before and after visible output.
- Default to avoiding transparent restart after output or tool execution has occurred.
- If restartable streams are supported, expose attempt/reset boundaries explicitly.
- Never silently replay tool effects.
- Correct the documentation.

Acceptance:
Midstream failure cannot silently duplicate caller-visible output or repeat side effects. Pre-output transient failures remain eligible for bounded retry.

6. The async tool runner does not await general awaitables

Location:
synaptic/core/_runner.py::run_tools_async

Observed:
A tool returning an asyncio.Future returned the Future itself as its result because only coroutine objects are awaited.

Required:
- Support general awaitables where the API promises async-compatible execution.
- Audit Flow, Pipeline and callback helpers for the same narrow coroutine checks.
- Ensure sync-only APIs reject async results cleanly without leaking unawaited coroutines.

Acceptance:
Coroutines, Futures and custom awaitables resolve to their values. Unsupported sync usage produces a clear error without resource warnings.

7. Tool binding permits duplicates within one batch

Location:
synaptic/core/model.py::Model.bind_tools

Observed:
Binding [tool, tool] to an empty model produces two entries with the same name.

Required:
- Deduplicate both against existing bindings and within the incoming batch.
- Define same-name replacement versus rejection explicitly.
- Support deliberate update/removal of tool definitions.
- Keep model, adapter and registry views consistent.

Acceptance:
Duplicate names cannot appear accidentally. Replacement/removal updates the schemas actually sent to providers.

**Source-observed limitations and improvements**

8. MCP hub lifecycle and scaling

Locations:
synaptic/mcp/hub.py
synaptic/mcp/session.py

Current limitations:
- load_all is sequential.
- unload_server removes tool registrations but does not close the connection.
- No explicit bounded worker/connection pool, idle eviction or circuit-breaker policy is visible in the inspected hub.
- Refresh removes old tools before completing registration of replacements.

Improve:
- Separate catalog visibility from connection ownership.
- Document unload, disconnect, remove and close independently.
- Add bounded concurrent startup and discovery.
- Isolate failures so one bad server does not block unrelated integrations.
- Add configurable concurrency limits, deadlines, idle eviction and reconnect backoff.
- Drain active calls during removal or replacement.
- Make registration refresh transactional from the caller's perspective.
- Keep eager loading optional.

Do not replace sequential execution with unbounded gather.

9. Tool identity and metadata preservation

Location:
synaptic/mcp/tools.py and registry integration

Current limitations:
- Tool names are formed as server_name + "__" + tool_name.
- The adapter primarily carries description and input schema.
- Prefix-based removal can become ambiguous with overlapping names.

Improve:
- Keep canonical identity separate from provider-facing function aliases.
- Make aliases collision-resistant, bounded and reversible.
- Track exact ownership for removal instead of ambiguous prefixes.
- Preserve output schemas and relevant annotations/metadata.
- Validate schemas and results at clear boundaries.
- Treat third-party behavioral annotations as claims, not authorization.
- Publish immutable or versioned tool snapshots for active turns.

Test naming collisions, overlapping prefixes, long names and schema replacement during use.

10. Registry isolation and subscription lifecycle

Location:
synaptic/core/tool.py and provider registry subscriptions

Current limitations:
- A process-global default registry exists.
- all() exposes the mutable internal dictionary.
- on_change() has no corresponding unsubscribe operation.
- Provider callbacks can retain adapter instances.
- Model/Agent construction can share mutable tools, history and instructions.

Improve:
- Make scoped registries the preferred explicit path.
- Preserve legacy global behavior only through a documented compatibility policy.
- Return read-only views or snapshots.
- Add disposable subscriptions and adapter/client cleanup.
- Define callback failure and reentrancy behavior.
- Document whether model instances may be shared concurrently; either support isolation or reject unsafe sharing.

Test cross-session isolation, direct-mutation prevention, callback cleanup and repeated adapter construction/disposal.

11. Tool schema generation and validation

Location:
synaptic/core/tool.py::autotool

Current limitation:
The inspected inference mostly maps primitive/container categories. It does not establish faithful nested schemas for richer Python annotations.

Improve:
- Resolve postponed annotations.
- Handle supported Optional/union, Literal, enum, nested list/dict and structured-model types.
- Define treatment of defaults, required fields and null.
- Reject unsupported annotations clearly instead of silently calling them strings.
- Validate arguments before invoking a tool.
- Validate declared outputs where practical.
- Use an established schema/type system where it simplifies correctness.

Test nested schemas, postponed annotations, invalid inputs and unsupported types.

12. Execution boundaries and extensibility

Locations:
Model autorun paths, Agent tool execution and core/_runner.py

Current limitations:
Agent owns direct tool execution, while permission support is a boolean callback. Consumers needing their own coordinator must carefully avoid competing execution paths.

Improve:
- Define a public executor interface that can be injected.
- Provide an explicitly documented proposal-only model interface.
- Guarantee autorun=False across sync, async and streaming paths.
- Allow external execution results to be supplied with stable call correlation.
- Keep the built-in executor useful for ordinary standalone agents.
- Ensure authorization hooks see the effective validated arguments, including defaults.
- Separate policy decisions from observational event handlers.
- Make permissive versus policy-required execution an explicit configuration choice.

Do not force Control's grant database or desktop policy into Synaptic.

Acceptance:
A consumer can own every tool effect while still using Synaptic's providers, streaming and optional agent orchestration.

13. Blocking synchronous work inside async paths

Locations:
run_tools_async, Flow.arun, Pipeline.arun and callbacks

Current limitation:
Sync callables are invoked directly on the event-loop thread.

Improve:
- Define which synchronous functions may execute inline.
- Offer explicit bounded thread/process offloading where appropriate.
- Preserve ordering and thread-affinity requirements.
- Document that cancelling a thread await does not necessarily stop the underlying function.
- Do not report cancellation as proof that no side effect happened.
- Make parallel tool execution opt-in or informed by explicit independence metadata.

Test event-loop responsiveness, bounded concurrency and cancellation of ongoing sync work.

14. Unified model/provider contracts

Locations:
core/base/base_model.py
core/_runner.py
core/_factory.py
providers/*

Current limitations:
- BaseModel formally requires only synchronous invoke, while async paths assume streaming support.
- ainvoke aggregates a stream.
- The inspected factory forwards constructor max_tokens only on the Claude branch.
- Completion and usage handling differs across paths.

Improve:
- Define explicit provider capability contracts for sync, async, streaming, tools, structured output, images and audio.
- Unsupported capabilities should fail clearly before starting work.
- Audit constructor/request configuration forwarding, especially token limits.
- Normalize finish reasons, usage, errors and tool-call correlation.
- Preserve provider-specific information through documented extension fields.
- Ensure assembled tool calls execute only once, after complete argument validation.
- Preserve native call IDs where required.
- Audit final-only output and interrupted-stream behavior.

Build a capability matrix. Do not claim identical support where providers differ.

15. Multimodal history and context integrity

Locations:
core/base/memory.py and provider history conversions

Current limitation:
The inspected memory API is text-centric even though invocation accepts images/audio.

Improve:
- Define a typed multimodal message representation.
- Preserve supported media references and tool call/result relationships through history serialization and provider conversion.
- Bound history by an explicit policy, with token-aware options where useful.
- Preserve complete tool exchanges when truncating context.
- Make repeated-tool-result compaction semantics explicit; identical arguments do not imply identical external state.
- Provide versioned serialization and external storage hooks where justified.

Keep persistence optional. A reusable framework need not own every consumer's durable action journal.

16. Run outcomes, cancellation and event behavior

Locations:
synaptic/agent/agent.py
synaptic/agent/session.py
synaptic/agent/events.py
synaptic/logging/*

Current limitations:
- _finish marks sessions done in finally, including failure/cancellation paths.
- Logging subscriber exceptions are swallowed.
- Agent event handler exceptions can interrupt execution.
- Registry and event subscriptions have incomplete lifecycle management.

Improve:
- Distinguish completion, failure, cancellation and turn/budget exhaustion.
- Define partial-history behavior on interruption.
- Propagate cancellation consistently and close owned resources.
- Separate observational callbacks from execution policy.
- Make observer failure handling explicit and configurable.
- Provide unsubscribe/disposal and bounded delivery where needed.
- Emit enough correlation metadata to reconstruct an operation's lifecycle.

Test observer failures before/after a tool effect, cancelled streams, exhausted turns and repeated session reuse.

17. Retry policy beyond streaming

Location:
synaptic/run/retry.py

Current limitations:
Defaults catch Exception broadly and use a fixed delay. Invalid retry counts are not clearly validated.

Improve:
- Validate retry counts and delay configuration.
- Distinguish transient transport/rate-limit failures from authentication, validation and programming errors.
- Support bounded exponential backoff, jitter, retry-after and total deadlines.
- Preserve original error context.
- Treat model generation retries separately from tool-effect retries.
- Do not automatically retry arbitrary side-effecting callables without explicit semantics.
- Keep the generic decorator available with clearly documented caller responsibility.

Test zero/negative retry configuration, cancellation during backoff and nonretryable failures.

18. Logging privacy, reliability and correlation

Locations:
synaptic/logging/_bus.py
synaptic/logging/_writer.py
logging calls in model/agent/MCP paths

Current limitations:
Events may include raw prompts, context, arguments and results. Logging state is global. Sink failures can disappear silently.

Improve:
- Add redaction/filter hooks before any sink receives data.
- Make payload logging explicit; default to useful metadata.
- Add session/run/call/attempt correlation where available.
- Preserve credential secrecy in endpoints, headers and errors.
- Define bounded asynchronous sink delivery and failure reporting.
- Support rotation/retention or delegate these through a documented sink contract.
- Distinguish best-effort diagnostics from reliable application-owned event recording.

Test seeded secrets, concurrent sessions, failing sinks and oversized payloads.

19. Packaging, optional dependencies and protocol compatibility

Locations:
pyproject.toml
synaptic/__init__.py
synaptic/providers/__init__.py
synaptic/core/_factory.py

Current limitations:
- No requires-python was declared in the inspected project metadata.
- Dependencies, including the MCP extra, lack meaningful version bounds.
- Provider imports are eager.
- No reviewed reproducible development lockfile was found.
- MCP compatibility is not established by the existing fake-session tests.

Improve:
- Declare supported Python versions.
- Keep core installation small.
- Move provider SDKs into appropriate extras and load them lazily.
- Report missing extras with actionable errors.
- Remove obsolete or redundant dependencies only after auditing usage.
- Use compatible dependency ranges for library consumers and locked environments for development/CI.
- Verify current MCP/SDK versions against official documentation at implementation time.
- Implement protocol behavior through the official SDK.
- Maintain an explicit tested compatibility matrix and handle unsupported extensions clearly.
- Separate credential identities when caching discovery or pooling connections.
- Clarify existing license metadata; do not silently relicense the project.

Acceptance:
Core imports work without every provider SDK. Installing one provider extra is sufficient for that provider. Real stdio and loopback HTTP fixtures pass against the declared supported protocol/SDK combinations.

**Implementation order**

1. Establish the actual baseline, preserve existing work, and record findings.
2. Add failing regressions for the seven reproduced defects.
3. Fix those defects in cohesive changes.
4. Stabilize public result, provider, executor and lifecycle contracts.
5. Improve MCP lifecycle, identity, discovery and bounded concurrency.
6. Improve isolation, cancellation, retries and logging.
7. Complete provider capability, multimodal and packaging work.
8. Add migration documentation and measured qualification evidence.

Investigate related defects discovered during implementation and fix them when they fit the affected boundary. Record larger unrelated work separately.

Avoid adding abstractions without a real consumer or acceptance test.

**Validation and completion requirements**

- Run deterministic tests without credentials or network access.
- Add real local stdio and loopback HTTP MCP integration fixtures.
- Keep live-provider tests explicitly configured and separate.
- Test startup failure, transport death, cancellation, reconnect and cleanup.
- Test concurrent sessions for tool/history/credential isolation.
- Test multimodal content and call/result correlation.
- Test uncertain side effects and retry boundaries.
- Test bounded resource use rather than claiming scalability from async syntax.
- Verify clean installation in the supported dependency configurations.
- Run appropriate formatting, type checks and regression suites.
- Record exact commands, results, limitations and compatibility versions.
- Update architecture, implementation rationale, issues and handoff records with the implementation.
- Provide an old-to-new API migration guide for breaking changes.
- Finish with a finding-by-finding report: fixed, improved, intentionally retained, or deferred with a concrete reason.

Do not report a feature as complete based only on mocks when its acceptance criteria require a real transport or provider.

The outcome should be a demonstrably better Synaptic: reliable standalone behavior, clear public contracts, and clean integration with applications that own their own execution and persistence.