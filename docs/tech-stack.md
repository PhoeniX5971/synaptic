# Technology and compatibility

Python library using setuptools, asyncio, provider SDKs and optional MCP SDK.
Inspected environment: Python 3.13.8; pytest 9.0.2; Pydantic 2.12.5;
OpenAI 2.16.0; Anthropic 0.84.0; google-genai 1.61.0; xai-sdk 1.8.1.
These are observed installed versions, not a supported-version matrix.

Project metadata currently lacks requires-python and meaningful dependency bounds;
`enum.StrEnum` and ExceptionGroup require Python 3.11+. Provider imports are eager.
MCP was absent at baseline. Packaging work (F-019) must establish compatible extras
and a reviewed development lock before declaring supported configurations.

LICENSE contains GPLv3 and README declares GPLv3; do not silently relicense.
