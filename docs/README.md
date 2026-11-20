# Synaptic Docs

This folder is the practical guide to Synaptic. Read in this order if you are
new to the framework:

1. [`quickstart.md`](quickstart.md)
2. [`Overview.md`](Overview.md)
3. [`core/model.md`](core/model.md)
4. [`core/tool.md`](core/tool.md)
5. [`core/agent.md`](core/agent.md)
6. [`core/memory.md`](core/memory.md)
7. [`core/streaming-events.md`](core/streaming-events.md)
8. [`core/flows.md`](core/flows.md)
9. [`core/ledger.md`](core/ledger.md)
10. [`providers/index.md`](providers/index.md)

Provider-specific notes:

- [`providers/openai.md`](providers/openai.md)
- [`providers/gemini.md`](providers/gemini.md)

Runnable examples live in [`examples/`](examples/).

## Reference

- [Runtime contracts](contracts.md): exact behaviour of execution, events, history and providers.
- [Provider matrix](provider-matrix.md) and [retry policy](retry-policy.md).
- [Migration notes](migration.md): changed public contracts.
- [Support and limitations](support.md): measured environments and known limits.
- [Release discipline](releasing.md), [CONTRIBUTING](../CONTRIBUTING.md),
  [SECURITY](../SECURITY.md), [CHANGELOG](../CHANGELOG.md).
