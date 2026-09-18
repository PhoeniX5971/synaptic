# Release discipline

Only the maintainer chooses versions, creates tags, publishes artifacts or changes
remote settings. Local qualification and the tracked CI workflow never publish.
The current project version remains 0.1.0; this work does not declare a new release.

Use semantic versioning as the release policy. During 0.x, incompatible public API or
serialized contract changes require a minor-version change and migration notes; patch
versions preserve documented public contracts. Once 1.0 is declared, incompatible
changes require a major version. Deprecations should be documented, warn when practical,
and remain for at least one minor release before removal, except urgent security fixes
whose compatibility impact must be documented. A future release must explicitly decide
whether it can read histories written by each supported prior serialization version.

Maintainer checklist:

1. Review [handoff](next-steps.md), [known issues](known-issues.md), [contracts](contracts.md)
   and [changelog](../CHANGELOG.md). Choose/version the final change; update migrations.
2. Confirm the private reporting route in [SECURITY.md](../SECURITY.md). This is currently
   an unmet public-release prerequisite; no contact or enabled channel has been invented.
3. From the intended revision, run the frozen commands in [CONTRIBUTING.md](../CONTRIBUTING.md):
   full runtime/typing gate and `scripts/check_release.py`. The latter verifies wheel/sdist
   markers, ten installs, positive/negative downstream typing and installed offline examples.
4. After the maintainer publishes the branch, require the tracked CI job to pass on the
   exact release revision. Configured CI is not proof of a remote run. Record the run URL.
5. Record artifact hashes, Python/platform/SDK versions, warnings and commands in
   [qualification](qualification.md). Review included files and license. Test proposed
   additional platforms/protocols/providers before expanding support claims.
6. Review the final changelog/version/tag/artifacts; the maintainer performs publication
   through their chosen release channel and verifies the resulting installed artifacts.

Current CI selects Ubuntu 24.04, CPython 3.13.8 and uv 0.12.10. It uses read-only repository
permissions, no provider credentials and immutable action commits. The pinned upstream
releases are [checkout v6.0.2](https://github.com/actions/checkout/releases/tag/v6.0.2) and
[setup-uv v8.0.0](https://github.com/astral-sh/setup-uv/releases/tag/v8.0.0). The configured
runner image and remote execution remain unqualified until that job actually runs.
Windows/macOS, other Python/SDK versions, live providers and scale remain outside the
current measured support boundary.
