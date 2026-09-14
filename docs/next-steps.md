# Next steps

Start T-006 by reproducing MCPHub refresh/removal ownership failures: overlapping
server-name prefixes, failed replacement registration, and an in-flight call while
unloading. Preserve a catalog until a complete replacement can publish atomically.

Then implement exact ownership and bounded hub startup/discovery, with explicit
unload/disconnect/remove/close semantics. Session primitives are qualified on MCP
1.30.0, but the hub is still sequential and prefix-based. Coordinate registry
subscription cleanup with adapter disposal rather than freezing merged tool views.

Read [qualification](qualification.md), [finding report](finding-report.md), and
[hardening brief](hardening-brief.md). T-007–T-009 follow in dependency order.
No publishing, remote/history changes or sibling-project modifications.
