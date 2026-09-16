# Known issues and measured limits

BUG-001–007 and original F-008–019 are implemented with evidence in the
[finding report](finding-report.md). The follow-up ownership/publication/policy failures
are resolved by test_followup, test_mcp_dispatch, test_hub_contract and
test_invocation_contract. Historical BUG-020 (stale ignored tests) remains resolved.

Control BUG-010/011 are resolved by the execution recovery/boundary/reconciliation suites;
see readiness-followup.md. Remaining qualification limits:

- **Typing boundary:** full-package mypy passes all 90 source files, with untyped bodies
  checked and no incremental cache. Pinned Vertex/xAI SDK imports lack typing metadata;
  only those namespaces have missing-import overrides. Dynamic SDK payloads are not claimed
  as strictly typed. The prior 81-error report is superseded by T-010.
- **Provider services:** no live provider/model was contacted. SDK HTTP/typed response
  fixtures qualify conversion/configuration only. Endpoint/model capabilities may be narrower.
- **Compatibility:** runtime suite/installs were measured on Python 3.13.8 Linux and pinned
  SDKs. Windows/macOS and other interpreter/platform/dependency combinations require qualification
  before support claims; package ranges do not establish it. No remote CI run is claimed.
- **Vertex SDK:** two protobuf deprecation warnings occur in its GenerationConfig fixture.
  They are recorded rather than globally suppressed. The two existing history truncation
  notices are expected behavior.
- **Cancellation:** a cancelled thread may keep running. Closed/cancelled streams preserve
  partial context; unresolved exchanges need consumer resolution. Per-call receipts and uncertainty survive serialization; external reconciliation cannot
  erase a receipt. No automatic effect replay or durable exactly-once guarantee.
- **Bounds:** discovery/media limits reject after decoding/reading up to their documented
  limits. They are not a hostile-network frame sandbox. Idle hub eviction runs on access.
- **History:** dedup_tools is opt-in lossy compaction; identical calls do not establish stable
  external state. Serializable results/extensions must be JSON-compatible. Resource references
  are preserved but currently rejected for provider submission.

No remaining implementation task from the accepted hardening briefs is represented as a
planned prerequisite. Live-service qualification, strict annotation expansion and additional platform
coverage are future release work, with no credentials/network authorization inferred here.
