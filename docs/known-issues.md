# Known issues and finding status

IDs map directly to the numbered brief. Historical “fixed” claims are superseded
only by current executable evidence. All seven defects reproduced before implementation (S-002).

| ID | Finding | Status / evidence or prerequisite |
| --- | --- | --- |
| BUG-001 | Lossy MCP normalization | investigating; early structured return discards blocks |
| BUG-002 | Discovery pagination | investigating; only first page consumed |
| BUG-003 | Failed connection readiness | investigating; existing-task path skips error check |
| BUG-004 | Async usage loss | resolved; completion parity fixtures include 12/3, zero and unavailable usage |
| BUG-005 | Stream retry replay | resolved; no restart after a yield or when autorun can cause effects |
| BUG-006 | General awaitables | resolved; awaitables and sync coroutine cleanup regressions |
| BUG-007 | Batch tool duplicates | resolved; explicit replacement/removal and provider schema regressions |
| F-008 | Hub lifecycle/scaling | planned T-006; depends on session correctness |
| F-009 | Canonical tool identity | planned T-006; prefix ownership remains ambiguous |
| F-010 | Registry isolation/subscriptions | planned T-006; mutable global state remains |
| F-011 | Rich schema validation | planned T-007; primitive inference insufficient |
| F-012 | Injectable executor/proposals | planned T-007; requires validated arguments |
| F-013 | Bounded offloading | planned T-008; sync work still runs inline |
| F-014 | Provider contracts | planned T-007; native differences need qualification |
| F-015 | Multimodal history | planned T-009; text-oriented memory remains |
| F-016 | Run outcomes/observers | planned T-008; finally still marks sessions done |
| F-017 | Retry policy | improved T-004; bounds and transient defaults fixed; backoff/jitter/deadline T-008 |
| F-018 | Logging privacy/reliability | planned T-008; disable payload logging for sensitive workloads |
| F-019 | Packaging/protocol support | planned T-009; eager SDK imports, unbounded dependencies |

## BUG-020 — stale ignored tests (resolved in T-001)

Baseline: 102 passed, 9 failed. Tests used removed `MemoryList`/`memoryList`, expected
`automem=False` to truncate history, or stubbed sync invoke for async aggregation.
Update tests to `entries`/`memory_list`, retained windows and async streams. This
reconciles tests to existing implementation; it does not resolve the seven defects.
