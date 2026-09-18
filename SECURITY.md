# Security reporting

This is a pre-1.0 framework with no promised security-maintenance window or response SLA.
The currently qualified development baseline and exact dependency/platform limits are
recorded in [qualification](docs/qualification.md). Version ranges do not establish
security support for every dependency release, platform or live provider.

A maintainer-approved private reporting route has not yet been designated. Before a
public release, the maintainer must choose, enable and verify one and update this file.
Do not send sensitive vulnerability details, credentials, transcripts or exploit payloads
to public issues. A public issue may request a private reporting route without disclosing
technical details. No email address or enabled private advisory channel is claimed here.

Once a private route is confirmed, reports should include affected versions, a minimal
sanitized reproduction, impact, relevant configuration and any proposed mitigation.
Coordinate disclosure with the maintainer; do not test third-party provider accounts or
systems without their authorization.

Synaptic executes tools authorized by its consumer; it is not a sandbox. Provider
credentials, execution grants, host isolation, durable effect recording and incident
response remain application responsibilities. Cancellation cannot undo an admitted
remote effect or running thread. See [contracts](docs/contracts.md) and
[known limitations](docs/known-issues.md) before deployment.
