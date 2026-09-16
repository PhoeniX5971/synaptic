# Provider capability and fixture matrix

All listed adapters implement sync invoke, async aggregation, streaming, tools and
structured JSON. Their native APIs, model restrictions and schema support differ.
`adapter.capabilities` describes the implemented transport contract; it does not
promise every model on an endpoint accepts every option. Unsupported media is rejected
by Model before dispatch. Select the appropriate model and provider configuration.

| Adapter | Extra / measured SDK | Images | Audio | Async stream | Call IDs / structured behavior |
| --- | --- | --- | --- | --- | --- |
| OpenAI | openai / 2.16.0 | yes | yes | native Chat Completions | native; JSON Schema |
| DeepSeek | deepseek / openai 2.16.0 | no | no | native Chat Completions | native; JSON object |
| Universal OpenAI | openai / 2.16.0 | transport supports | transport supports | native Chat Completions | native; JSON object; endpoint decides support |
| Claude | claude / anthropic 0.84.0 | yes | no | native message events | native; forced structured-output tool |
| Gemini | gemini / google-genai 1.61.0 | yes | yes | native | native when supplied; generated otherwise; response schema |
| Vertex | vertex / google-cloud-aiplatform 1.71.1 | no | yes | native generate_content_async | local stable IDs; response schema/JSON MIME |
| Together | together / 2.2.0 | transport supports | transport supports | bounded thread bridge | native; JSON object |
| xAI | xai / 1.8.1 | yes | no | native gRPC | native; JSON Schema |

Constructor max_tokens and per-call overrides are forwarded on sync/async paths.
Chat providers preserve native IDs across history; Vertex's measured legacy API has
no native ID field. The generic resource ContentBlock is retained in serialized
history and rejected for provider submission. xAI raw audio is explicitly unsupported;
a file upload alone is not an established audio message contract.

Evidence: test_provider_sdk_contract.py uses real OpenAI SDK serialization/parsing
and httpx.MockTransport for OpenAI, DeepSeek and universal endpoints.
test_provider_parity.py exercises installed SDK response/event types and captured
requests for Google, Anthropic, Vertex and xAI; Together's blocking iterator bridge
uses the compatible Chat response fixture. test_history_contract.py and the retained
test_adapters.py exercise history conversion. These are SDK/adapter fixtures, not
live-service or model qualification. No credentials or provider network requests
were used. Future SDK versions inside dependency ranges remain unqualified until rerun.
