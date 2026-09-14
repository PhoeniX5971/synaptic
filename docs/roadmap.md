# Roadmap

Objective: reliable reusable model and agent orchestration, with consumers free to
own execution and persistence. Scope and acceptance: [brief](hardening-brief.md).

| ID | Milestone | Status | Exit evidence |
| --- | --- | --- | --- |
| M-001 | Establish baseline and production records | done | S-001; existing work preserved, tests tracked |
| M-002 | Reproduce and fix seven reviewed defects | active | T-002–T-005; regression and transport evidence required |
| M-003 | Public executor/provider/lifecycle contracts | planned | Findings 8–14, 16; boundary tests |
| M-004 | Isolation, retries, logging, multimodal history | planned | Findings 10, 13, 15–18; concurrency/privacy tests |
| M-005 | Packaging and qualification | planned | Finding 19; install matrix and real transport fixtures |

No release-readiness claim follows from the historical hardening commit.
