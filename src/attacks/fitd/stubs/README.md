# Pinned garak typing boundary

These declarations cover the garak symbols used by the FITD integration at
revision `3f50ea5ff9cd7050099940647c15c39b07a93392`. They are a used-surface
description, not a complete replacement for garak's public API. Adding another
native operation requires checking its implementation and extending the relevant
declaration.

Pyright's `stubPath` must point to `src/attacks/fitd/stubs`. Runtime imports use
the installed garak package; this directory contains no runtime replacement.
`../native.py` statically imports the original classes, and `../native.pyi`
re-exports those same declared classes.

The declarations retain the real probe inheritance chain:
`FITD → IterativeProbe → Probe → Configurable`. They also retain native nullable
message text and prompts, free-form attempt notes, and optional verification
results. The English FITD adapter validates text, prepared prompts, and integer
ladder progress at the boundary. No `Any`, catch-all attributes, diagnostic
suppression, or flattened surrogate hierarchy is used.

The private provider/generator protocols in `probes/base.pyi` describe the
operations called by the native loop. They are typing helpers, not additional
runtime exports from garak.
