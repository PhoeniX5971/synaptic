# Technology and compatibility

Python >=3.11 library using setuptools, asyncio and optional provider/MCP SDKs.
Core dependencies: Pydantic >=2.12,<3 and jsonschema >=4.26,<5. Provider extras are
openai, deepseek, claude, gemini, vertex, together and xai; mcp is separate, all combines
them. SDKs load lazily. dotenv is no longer a direct dependency or auto-loaded by adapters.
Legacy google-generativeai/vertexai umbrella distributions were replaced by the SDKs
actually imported: google-genai and google-cloud-aiplatform.

`uv.lock` pins development/CI resolution and all extras. `uv sync --frozen --all-extras`
recreates it; library consumers use bounded ranges. Python 3.13.8 Linux and the SDK
versions in qualification.md are measured. The Python floor follows StrEnum,
ExceptionGroup, TaskGroup and asyncio.timeout; other runtimes are not claimed as tested.

LICENSE remains the existing GPLv3 text and is included in project metadata/distributions.
No license grant was changed. Provider-specific behavior and qualification limits are in
provider-matrix.md; the official MCP SDK 1.30.0 / protocol 2025-11-25 transport path is the
only measured MCP combination. No version-range inference substitutes for qualification.

The development typing gate is mypy 1.19.1 plus types-jsonschema 4.26.0.20260518,
configured in pyproject.toml and driven by scripts/check_quality.py. All runtime-package
files and untyped bodies are checked; only the Vertex/xAI SDK import metadata limitations
are exempt. Windows and macOS have not been qualified; the install harness is POSIX.

Inline typing is published with py.typed in wheel/sdist. Installed consumer contracts are
checked independently in core-only and selected extra environments. The tracked Linux
CI baseline uses CPython 3.13.8 and uv 0.12.10; remote execution remains unverified.
