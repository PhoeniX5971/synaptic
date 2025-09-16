# Overview

Synaptic is a minimal framework for orchestrating LLMs and developer-defined
tools with a small, testable core.

## Key Concepts

- **Memory** — typed objects representing turns / model responses.
- **History** — sliding-window of `Memory` objects used to build prompt context.
- **Tool** — a callable function made available to the model via provider function-calling.
- **Model** — provider-agnostic wrapper that dispatches to provider adapters (Gemini/OpenAI).

## Quick links

- Core:
  - [`docs/core/memory.md`](https://github.com/PhoeniX5971/synaptic/blob/main/docs/core/memory.md)
  - [`docs/core/model.md`](https://github.com/PhoeniX5971/synaptic/blob/main/docs/core/model.md)
  - [`docs/core/tool.md`](https://github.com/PhoeniX5971/synaptic/blob/main/docs/core/tool.md)
- Providers:
  - [`docs/providers/gemini.md`](https://github.com/PhoeniX5971/synaptic/blob/main/docs/providers/gemini.md)
  - [`docs/providers/openai.md`](https://github.com/PhoeniX5971/synaptic/blob/main/docs/providers/openai.md)
- Examples:
  - [`docs/examples`](https://github.com/PhoeniX5971/synaptic/blob/main/docs/examples)
