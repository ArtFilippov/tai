# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`.
- Spec: `.scratch/<feature-slug>/spec.md`.
- Implementation issues: `.scratch/<feature-slug>/issues/<NN>-<slug>.md`,
  one file per ticket, numbered from `01`.
- An implementation issue's triage state is a `Status:` line near the top of the file.
  Role strings are defined in `docs/agents/triage-labels.md`.
- Append comments and conversation history to the bottom of the file under `## Comments`.

## Tracker operations

- "Publish to the issue tracker": create a file using the conventions above,
  creating its directory if needed.
- "Fetch the relevant ticket": read the file at the referenced path or issue number.
  Resolve numbers within the relevant feature directory;
  if the number is ambiguous, ask for the path.

## Wayfinding

For `/wayfinder`, the map and child tickets live in separate files:

- Map: `.scratch/<effort>/map.md`, with Notes / Decisions-so-far / Fog sections.
- Child ticket: `.scratch/<effort>/issues/<NN>-<slug>.md`,
  numbered from `01`, with the question in the body.
- `Type:` records the ticket type: `research`, `prototype`, `grilling`, or `task`.
- For wayfinder child tickets, `Status:` records the work stage:
  `open`, `claimed`, or `resolved`.
- Blocking: a `Blocked by: NN, NN` line near the top.
  A ticket is unblocked when every listed ticket has status `resolved`.
- Frontier: the first ticket by number with status `open` and no unresolved blockers.
- Claim: save `Status: claimed` before starting work.
- Resolve: append the answer under `## Answer`, set `Status: resolved`,
  then append a brief summary and link to Decisions-so-far in the map file.
