---
name: dedupe-assembly
description: Use when you notice repeated object construction or dependency wiring while coding in a Quard project.
---

# Deduplicate Object Assembly

Use this as a design prompt, not a required sequence of wrappers. Each choice must solve a distinct problem in the code at hand.

1. **Extract the responsibility.** Compare the repeated construction sites: what behavior is shared, what inputs vary, and who owns each resulting object's lifetime? Put the shared behavior in one object with an explicit local contract. Keep different per-call or per-run composition at the assembly sites. Name an abstraction for what it actually owns; a model-specific dependency does not become generic merely because its owner has a generic name.

2. **Consider an Envelope.** When a variant must keep an existing contract while adding behavior or composing its delegate, use an Envelope. Reuse an existing base Envelope if it already forwards that contract. Under Quard, a base Envelope stores one injected delegate and forwards operations unchanged; an assembly Envelope has only a constructor that calls `super().__init__(...)`. Put resource loading or other work in a separate object behind that forwarding interface. Read `quard/PROFILE.md` (Contract folders) for placement and delegation rules before adding an Envelope.

3. **Consider a lambda factory.** If an object needs a factory rather than a finished dependency, inject a lambda that captures configuration or accepts dependencies as parameters. The receiving object invokes the factory when it needs the dependency. Invoking it on first use enables lazy creation; caching can then reuse the result. Keep the expression small and its captured values proven at the assembly site; Quard's lambda-injection check restricts both shapes and origins. Read `quard/PROFILE.md` (Lambda injection) when wiring it. Use an object with a contract when the construction expression needs more than a simple lambda.

4. **Consider caching.** If repeated use by the *same owner* must reuse the factory's result, cache that owner's deferred value. Keep the cache behind the Envelope when Quard requires the base Envelope to forward directly. `qraft.cached_property` provides a read-only per-instance cache on first access; put `@cached_property` above `@override` when implementing a getter-only contract property, and pin `qraft` as a runtime dependency as described in `quard/README.md` (Optional runtime: qraft). A failed computation can be retried; a requirement for exactly one concurrent first computation needs separate synchronization. Choose when validation runs along with when the dependency is created; it may also be deferred.

5. **Verify the chosen behavior.** Check the relevant construction sites, factory invocation count, cache lifetime, and timing of work with focused tests. If Python changed, run the project's required runtime tests and full Quard check; report its actual exit code and unresolved diagnostics. Finish when shared construction logic has one owner and each added Envelope, factory, or cache has a specific reason to exist.
