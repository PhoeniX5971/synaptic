# Control integration review follow-up (T-011)

Reviewed Control's `docs/synaptic-integration-review.md` read-only against clean Synaptic
`3eb9956d9d9441dedf977c2a6ee76696e1ab42c9` on 2026-09-18. Control's BUG-012–015 IDs belong
to that review; earlier Synaptic tasks/findings are preserved.

| Finding | Resolution | Public integration evidence |
| --- | --- | --- |
| BUG-012: Retry blocked by its own automem failure | Independent failed-attempt records, attempt/output ownership, guarded backoff | test_public_integration and test_public_retry_integration: real Model + Retry, automem on/off, sync/async/stream, repeated failure, output/effect barriers, unresolved history, deadlines/cancellation |
| BUG-013: growing receipts bypass context budgets | Enforce after publication/reconciliation and before projection/submission; independent recovery retention | test_public_budget_integration: Model/Agent modes, oversized receipts, partial failure, external completion, restored recovery and last-moment growth |
| BUG-014: contradictory IDs permit receipt replacement | Atomic format/identity validation and canonical known-result indexing | test_public_reconciliation: mapping/list/legacy forms, malformed inputs, immutability and restored histories |
| BUG-015: installed Model becomes Any | py.typed in both artifacts, typed exports/returns and signature-preserving ownership decorator | check_consumer_types: valid assert_type usage and all 12 deliberately invalid lines checked in core-only, OpenAI and all-extras wheel environments |
| Public release entry points | Tracked pinned CI, contribution, security and release guidance, changelog, installed offline example | check_quality/check_release; remote run and private reporting route remain maintainer prerequisites |

All four initial runtime acceptance cases failed before implementation, then passed.
The first installed-type run correctly rejected invalid code but the harness expected
absolute diagnostic paths; accepting mypy's relative filenames corrected that harness,
without changing/suppressing the intended type errors. See qualification.md for final
fresh-environment counts and distribution results. Earlier 302 tests remain intact.

No live service, Windows/macOS, additional Python/SDK/protocol, throughput or remote CI
qualification is asserted. The framework retains in-memory evidence; consumers still
own durable effects, storage, grants and retention. SECURITY.md deliberately does not
invent a reporting address or claim private advisories are enabled.
