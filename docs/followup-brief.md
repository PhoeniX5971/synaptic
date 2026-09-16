Complete Synaptic’s remaining hardening work and fix the failures identified in this follow-up review.

Reviewed baseline:
6a8b98701b0f3cba95ac2ea7e16d4a1d44fe7b0f

The existing suite passes 176 tests with two expected history-truncation warnings. The seven original regression cases pass. Preserve those improvements.

The original report is only partially implemented. docs/finding-report.md correctly describes the first tranche; the remaining planned work still needs implementation and qualification.

Inspect the current checkout and preserve unrelated changes. Use the production and handoff skills, maintain project records, document breaking changes, and commit cohesive changes. Do not publish or modify sibling projects.

**Fix these reproduced failures first**

1. An old request failure can destroy a replacement connection

Location:
synaptic/mcp/session.py::MCPSession.call_tool

Reproduction:
- Connect session/owner A.
- Start a tool call that waits on a fixture gate.
- Reconnect successfully to session/owner B.
- Release A’s old call with ConnectionError.

Actual:
B changes from connected to failed, its session becomes None, and its owner task is cancelled.

Cause:
The exception handler invalidates the currently stored session/task without checking which connection dispatched the failed request.

Required:
Capture dispatch ownership: connection generation/attempt, SDK session and owner task. A late failure may invalidate only its matching owner. Preserve the old request’s error and uncertain effects without corrupting B.

Regression tests:
Old success, failure and timeout after reconnect leave B usable. Failure of the current owner still invalidates it correctly. Verify cleanup and concurrent requests.

2. Unloading one server removes another server’s tools

Location:
synaptic/mcp/hub.py::MCPHub.unload_server

Reproduction:
- Load server "a" with tool "one".
- Load server "a__b" with tool "two".
- Registry contains "a__one" and "a__b__two".
- Unload "a".

Actual:
Both registrations disappear, while loaded still reports "a__b".

Required:
Replace prefix-based deletion with exact registration ownership. Introduce collision-safe canonical identities and provider aliases. Keep registry and loaded state consistent.

Regression tests:
Overlapping server names, cross-server alias collisions, independent unload, refresh and removal cannot delete or rebind another instance’s tools.

3. Failed refresh destroys the previous hub catalog

Locations:
synaptic/mcp/hub.py::MCPHub.load_server
synaptic/core/tool.py registry notification methods

Reproduction:
- Load a valid catalog.
- Register an on_change observer that raises RuntimeError.
- Refresh the same valid catalog.

Actual:
Refresh raises and leaves both the registry and loaded state empty.

Session-level atomic discovery does not make hub publication atomic.

Required:
Prepare and validate replacements before mutation. Publish registry and ownership state atomically. Define observer failure independently from publication success.

Also audit MCPHub.close: it currently unregisters tools before closing sessions. An unregister observer failure must not prevent cleanup. This cleanup consequence needs its own regression test.

Regression tests:
Failed preparation retains the old catalog; notification failure leaves consistent state; shutdown attempts every owned close; readers never observe partially published replacements.

4. Permission checks authorize different arguments from execution

Locations:
synaptic/agent/agent.py::_tool_results / _atool_results
synaptic/core/tool.py::_run_sync / _run_async

Reproduction:
- Configure a tool default: path="outside-approved-scope".
- Propose the tool with arguments {}.
- A recording permission callback approves the supplied {}.
- The tool subsequently merges defaults.

Actual:
Policy checks {}, but the function receives the outside path.

The reproduction used an in-memory recording function, not an actual file write. This demonstrates inconsistent policy inputs, not that every consumer permits missing arguments.

Required:
Resolve tool identity, merge defaults and validate into one immutable effective invocation before authorization. Execute exactly the authorized invocation.

Prevent observers, policy callbacks or concurrent mutations from changing checked arguments. Specify which defaults callers may override.

Regression tests:
Forbidden default targets are rejected before effects. Valid defaults are checked. Executed arguments match authorized arguments across sync, async, streaming and external-executor paths.

**Finish the outstanding original findings**

8. Hub lifecycle/scaling:
Bounded concurrent loading, failure isolation, explicit unload/disconnect/remove/close semantics, draining and resource limits.

9. Tool identity/metadata:
Exact ownership, collision-safe aliases, preserved schemas/metadata and stable tool snapshots.

10. Registry isolation:
Scoped registries, protected snapshots, disposable subscriptions and adapter cleanup.

11. Rich schemas:
Faithful supported Python annotation inference, explicit unsupported-type errors and effective argument validation.

12. External executor:
Injectable execution boundary, reliable proposal-only integration and correlated external results.

13. Async offloading:
Explicit bounded scheduling for blocking synchronous work, respecting thread affinity and cancellation/effect limitations.

14. Provider contracts:
Capability matrix, configuration forwarding—including token limits—native call IDs, completion metadata and provider-specific parity fixtures.

15. Multimodal history:
Typed media preservation, complete tool exchanges, safe truncation and versioned serialization.

16. Outcomes and observers:
Distinct success/failure/cancellation/budget outcomes; defined observer failure handling and resource cleanup.

17. Retry policy:
Retain the stream-replay fix. Complete transient error classification, retry-after, bounded backoff/jitter and total deadlines without replaying uncertain effects.

18. Logging:
Redaction before sinks, scoped correlation, explicit payload logging, bounded delivery and visible sink failures.

19. Packaging/protocol:
Provider extras and lazy imports, requires-python, reproducible development dependencies, clean-install matrix and explicitly tested compatibility.

Current measured MCP support is SDK 1.30.0 with the tested legacy protocol path. Do not infer broader support from dependency ranges or describe untested versions as qualified.

**Completion criteria**

- Preserve the existing 176 passing tests.
- Add failing regressions before fixing the four cases above.
- Re-run real local MCP transports.
- Test isolation, effective-argument equality, cancellation and cleanup.
- Verify core imports without unrelated provider SDKs and clean installation of declared extras.
- State static-check and provider-qualification limits accurately.
- Update architecture, rationale, issue status, migration notes and handoff records.
- Report every original finding as implemented with evidence, or explicitly excluded by user-selected scope.

A planned prerequisite is scheduling information, not completion evidence. Finish the remaining scope rather than treating another partial tranche as completion of the original request.

Keep Synaptic reusable. Control-specific desktop automation, application grants and PC scheduling remain outside the framework.