# Known issues and measured limits

BUG-001–007 and original F-008–019 are implemented with evidence in the
[finding report](finding-report.md). The follow-up ownership/publication/policy failures
are resolved by test_followup, test_mcp_dispatch, test_hub_contract and
test_invocation_contract. Historical BUG-020 (stale ignored tests) remains resolved.

Remaining qualification limits:

- **Typing:** whole-project mypy reports 81 errors in 14 files, including incomplete
  optional annotations, dynamic mixin attributes and provider SDK types. This is a measured
  limitation, not a claim that every error is inherited. Ten selected core/MCP modules pass.
- **Provider services:** no live provider/model was contacted. SDK HTTP/typed response
  fixtures qualify conversion/configuration only. Endpoint/model capabilities may be narrower.
- **Compatibility:** runtime suite/installs were measured on Python 3.13.8 Linux and pinned
  SDKs. Other interpreter/platform/dependency combinations are not qualified by package ranges.
- **Vertex SDK:** two protobuf deprecation warnings occur in its GenerationConfig fixture.
  They are recorded rather than globally suppressed. The two existing history truncation
  notices are expected behavior.
- **Cancellation:** a cancelled thread may keep running. Closed/cancelled streams preserve
  partial context; unresolved exchanges need consumer resolution. No automatic effect replay.
- **Bounds:** discovery/media limits reject after decoding/reading up to their documented
  limits. They are not a hostile-network frame sandbox. Idle hub eviction runs on access.
- **History:** dedup_tools is opt-in lossy compaction; identical calls do not establish stable
  external state. Serializable results/extensions must be JSON-compatible. Resource references
  are preserved but currently rejected for provider submission.

No remaining implementation task from the accepted hardening briefs is represented as a
planned prerequisite. Live-service qualification, typing expansion and additional platform
coverage are future release work, with no credentials/network authorization inferred here.
