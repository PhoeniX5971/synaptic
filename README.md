# Synaptic

> A modular framework for building, managing, and running AI models and tools
> with provider abstraction.

---

## Features

- **Core abstractions**
  - `base_model` – shared model logic
  - `memory` – memory handling utilities
  - `tool` – tool integration layer
- **Provider support**
  - OpenAI (`synaptic.providers.openai_`)
  - Gemini (`synaptic.providers.gemini`)
- **Extensible**: Easily add new providers or tools by extending the core base classes.
- **Lightweight**: Minimal dependencies, designed for composability.

---

## Project Structure

```

synaptic/
├── core/               # Core framework modules
│   ├── base/           # Base abstractions
│   ├── model.py        # Generic model management
│   ├── provider.py     # Provider handling
│   └── tool.py         # Tool integrations
├── providers/          # AI provider implementations
│   ├── openai\_/        # OpenAI provider
│   └── gemini/         # Gemini provider
├── main.py             # Entry point (if running standalone)

```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/synaptic.git
cd synaptic

# Install dependencies
pip install -e .
```

---

## Wiki

[`Docs`](https://github.com/PhoeniX5971/synaptic/blob/main/docs/)
