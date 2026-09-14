# Development plan

| Task | Status | Work and gate |
| --- | --- | --- |
| T-001 | done | Preserve baseline and establish canonical records; S-001 |
| T-002 | done | Reproduce BUG-001–007 against current tree before fixes |
| T-003 | done | Preserve usage; awaitables; explicit binding replacement/removal |
| T-004 | done | Prevent streamed replay and validate retry configuration |
| T-005 | active | Lossless MCP results, bounded discovery and owner-task lifecycle |
| T-006 | planned | Hub concurrency/ownership/identity and scoped registry cleanup |
| T-007 | planned | Validated schemas, injected executor and provider capabilities |
| T-008 | planned | Cancellation outcomes, history isolation, offloading and private logging |
| T-009 | planned | Multimodal history, optional SDKs, install matrix and migration qualification |

Follow the brief's implementation order. First tranche is T-001–T-005. Later
milestones remain planned, not silently accepted or marked complete. Add related
boundary regressions while fixing defects; keep unrelated expansion in this plan.
