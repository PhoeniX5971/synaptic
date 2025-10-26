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
│   ├── openai\_/       # OpenAI provider
│   └── gemini/         # Gemini provider

```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/PhoeniX5971/synaptic.git
cd synaptic

# Install dependencies
pip install -e .
```

---

## Wiki

[Synaptic Wiki](https://synaptic-wiki.vercel.app)

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3).

Copyright (c) 2025 PhoeniX5971

You are free to use, modify, and distribute this project under the same license.
Please retain author attribution in redistributions.

---

### Disclaimer

This software is provided "as is", without warranty of any kind, express or implied,
including but not limited to the warranties of merchantability, fitness for a particular
purpose, and noninfringement.

In no event shall the author or contributors be liable for any claim, damages, or other
liability, whether in an action of contract, tort, or otherwise, arising from, out of,
or in connection with the software or the use or other dealings in the software.
