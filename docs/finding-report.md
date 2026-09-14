# First-tranche finding report — 2026-09-14

The user's request was to clean up and start the brief. T-001–T-005 are complete;
this report does not mark the full improvement program complete. Evidence and
version limits: [qualification](qualification.md). API changes: [migration](migration.md).

| Finding | Disposition | Evidence or concrete remaining work |
| --- | --- | --- |
| 1 — MCP results | fixed | Mixed real stdio output; structured JSON round trips, media/resource metadata, explicit legacy helper, distinct call errors |
| 2 — Pagination | fixed | Real two-page stdio catalog; count/byte/time/cursor/schema failure tests preserve prior session catalog |
| 3 — Session lifecycle | fixed | Coalesced startup failure, cancellation, timeout, slow-close quarantine, reconnect, EOF and process cleanup tests |
| 4 — Usage loss | fixed | Sync fixture, ainvoke, streamed history and Agent records agree for 12/3, zero and unavailable usage; native SDK conversion differences remain finding 14 |
| 5 — Stream replay | fixed | Error propagates after first yield; unsafe/unknown-effect producers do not retry by default; bounded pre-output generation retries remain |
| 6 — Awaitables | fixed | Coroutines, Futures, custom awaitables; Flow/Pipeline/permission/events; sync rejection closes coroutines |
| 7 — Tool duplicates | fixed | Duplicate batch entries deduplicate; conflicting local names reject; explicit replacement/removal updates actual provider schemas |
| 8 — Hub scaling/lifecycle | deferred T-006 | Needs exact ownership and transactional registry update before bounded concurrent hub loading; session primitives now ready |
| 9 — Tool identity | deferred T-006 | Prefix aliases/removal remain; canonical identity must ship together with ownership transactions |
| 10 — Registry isolation | deferred T-006 | Model sees current registry changes, but read-only registries/disposable subscriptions require adapter cleanup changes |
| 11 — Rich schemas | deferred T-007 | Needs shared validation boundary used by executor and adapters; MCP input-schema validation alone does not solve Python annotation inference |
| 12 — External executor | deferred T-007 | Authorization must receive validated defaults; changing execution ownership depends on finding 11 |
| 13 — Offloading | intentionally retained inline pending T-008 | Thread affinity and uncertain effects need an explicit bounded scheduling contract before moving synchronous work |
| 14 — Provider contracts | improved / deferred T-007 | Shared completion metadata added; native capability/configuration/call-ID audit needs provider-specific fixtures, not an invented parity claim |
| 15 — Multimodal history | deferred T-009 | Lossless MCP output is not multimodal conversation history; needs provider conversion and tool-exchange truncation design |
| 16 — Outcomes/observers | deferred T-008 | Session-finally behavior and observer failure policy need an execution-lifecycle change; awaitable cleanup alone is insufficient |
| 17 — Retry policy | improved | Counts/delays validated, transient wrapper defaults and effect boundaries; SDK retry-after/backoff/jitter/deadlines remain T-008 |
| 18 — Logging | deferred T-008 | Preserved user implementation and local raw logs; privacy filters/scoping/bounded sinks require a separate event/sink contract |
| 19 — Packaging/protocol | improved | MCP range bounded and real SDK 1.30.0/stdin/HTTP qualified; optional provider imports, locks, clean install and Python matrix remain T-009 |

Retained compatibility is documented rather than silently removed: legacy global
registry, generic caller-responsible retry decorator, explicit lossy MCP helper,
and optional application-owned persistence. No live providers were contacted.
