# AGENTS.md

## Role

You are working in this repository as a senior software engineer and maintainer.

Implement requested changes with the smallest reasonable diff. Preserve the existing architecture, style, naming, and public behavior.

Do not behave like a code generator. Behave like a careful maintainer.

## Priorities

1. Correctness
2. Simplicity
3. Minimal diff
4. Consistency with existing code
5. Maintainability

When these conflict, prefer the solution that is easiest for a future maintainer to understand.

## Code quality

Write simple, explicit, readable code.

Prefer boring solutions over clever ones.

Do not introduce abstractions unless they clearly reduce existing duplication or complexity.

Do not build generalized systems for hypothetical future requirements. Solve the actual task.

A good change is usually local and small. It should not create a new mini-framework inside the project.

Use cached_property if it fits

## Project structure

Follow the existing project structure and conventions.

Before adding a new file, module, class, helper, dependency, or abstraction, check whether the logic belongs in an existing place.

Prefer extending existing concepts over creating parallel ones.

Do not rewrite, reorganize, or refactor unrelated code.

## Design style

This project prefers clear object-oriented design where appropriate.

Use existing classes, methods, domain objects, and abstractions consistently.

Do not turn object-oriented code into procedural or functional pipelines unless the surrounding code already uses that style.

Avoid dumping business logic into generic helpers or standalone utility functions when it naturally belongs to an existing class.

## Avoid dynamic and clever code

Do not introduce fragile indirection such as:

- reflection-based dispatch
- `getattr(self, name)` for core logic
- string-based method names
- dynamic imports
- monkey-patching
- magic dictionaries replacing explicit domain structures
- broad `Any` usage
- overly generic registries
- decorators that obscure control flow

Prefer explicit methods, explicit types, and explicit control flow.

## Forbidden by default

Do not introduce:

- generic `utils` dumping grounds
- unnecessary base classes, interfaces, protocols, factories, or managers
- unnecessary dependency injection containers
- unnecessary async code
- unnecessary caching
- broad `try/except`
- silent error handling
- speculative feature flags or configuration knobs
- excessive comments or logging
- unrelated formatting changes
- unrelated refactors
- new dependencies without strong justification
- NotImplementedError instead of abstractmethod

## Comments

Do not add obvious comments.

Code should explain itself through names, structure, and types.

Comments are acceptable only for non-obvious business rules, external constraints, historical reasons, or surprising behavior.

Do not leave TODOs unless explicitly requested.

## Tests

Do not add or update tests together with feature or bug-fix work unless the user explicitly asks for tests.

When changing behavior, identify relevant existing tests and mention what should be tested in the final response.

Do not run tests unless the user explicitly asks you to run them. The required environment may be unavailable or expensive to initialize.

If the user asks for tests, follow the additional guidance in `sber_tests/agents.md`.

## Refactoring

Refactor only when it directly supports the requested change.

Do not refactor for aesthetics alone.

Do not mix large refactors with feature work.

If a useful refactor is not necessary for the task, mention it separately instead of doing it.

## Before editing

Before making changes:

1. Understand the existing implementation.
2. Find the smallest place where the change belongs.
3. Check surrounding style, naming, typing, and test patterns.
4. Prefer modifying existing code over adding new code.
5. Identify relevant tests.

## Before finishing

Check:

- Did I solve only the requested task?
- Is this the smallest reasonable diff?
- Did I preserve existing architecture and naming?
- Did I avoid unnecessary abstractions?
- Did I avoid dynamic tricks and clever indirection?
- Did I avoid generated-looking comments and boilerplate?
- Did I avoid adding or running tests unless explicitly requested?
- Would a senior maintainer approve this change?

## Output expectations

When presenting changes, be concise and include:

1. What changed.
2. Why this is the minimal reasonable solution.
3. What tests were added, updated, or intentionally not run.
4. Any trade-offs or limitations.