# Context

## Steps

### Context

1. Create a data-context report using existing descriptions and reports first.
   - Record provenance, shape, variable semantics, granularity, and temporality.
   - Record missingness; determine and document handling appropriate to the data and visualization goal.
   - Record known data caveats, accepted assumptions, and relevant unknowns.

2. If no context search location has been defined, follow [Placement](#placement).

3. Save the report following the defined placement rules and return its path.

**Done:** The persisted report identifies the available sources and scope. Every applicable context item, including missingness handling, is recorded or marked unknown. Unknowns block completion only when they prevent the requested task.

### Placement

1. Ask the user whether the default placement is suitable; use it on acceptance or use the alternative location they specify.

2. Record the chosen search location in an existing applicable `AGENTS.md`, or in `README` when no `AGENTS.md` applies.
   - For default placement, use: `Data contexts are stored in the default location.`
   - For custom placement, give the path where contexts should be searched for.

**Done:** The context search location is defined and discoverable from project documentation.
