# Domain Docs

This repo uses a single-context layout.

## Before exploring the project

- Read root `CONTEXT.md` for the domain vocabulary and model.
- Read documents in `docs/adr/` that relate to the current task
  for existing architectural decisions.

If these files or directories do not exist, proceed silently.
The `/domain-modeling` skill creates them lazily as terms and decisions are resolved,
including through `/grill-with-docs` and `/improve-codebase-architecture`.

## Use the glossary's vocabulary

In issue titles, refactor proposals, hypotheses, and test names,
use the terms defined in `CONTEXT.md`, respecting the synonyms it explicitly avoids.

If a concept is missing from the glossary, check whether it belongs to the project's language.
Note a genuine gap for `/domain-modeling`.

## Flag ADR conflicts

If a proposal contradicts an existing ADR, explicitly identify that document
and explain why the decision is worth revisiting.
