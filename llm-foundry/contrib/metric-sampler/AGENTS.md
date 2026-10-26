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

Use cached_property if it fits.

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
- any mention of AI assistants or their vendors in code, comments, commits, configs or run names

## Comments

Do not add obvious comments.

Code should explain itself through names, structure, and types.

Comments are acceptable only for non-obvious business rules, external constraints, historical reasons, or surprising behavior.

Do not leave TODOs unless explicitly requested.

## Tests

Tests live in `sber_tests/unit` and mirror the package layout. Every test module sets `pytestmark = pytest.mark.unit`.

Coverage must be 100% (branch coverage) and tests must be meaningful: assert on behavior and full values, not on call order or private details. Mock only at repository boundaries (chdb sessions, filesystem outside `tmp_path`, S3, subprocess). Prefer parametrization over copy-paste.

Do not run tests that need the cluster. Unit tests must run on a laptop without chdb installed.

## Secrets

Credentials are read from files on the host (rclone.conf, env files) at runtime and never printed, logged, echoed into shell commands, or written into test fixtures as realistic values.

## Refactoring

Refactor only when it directly supports the requested change.

Do not mix large refactors with feature work.

If a useful refactor is not necessary for the task, mention it separately instead of doing it.

## Before finishing

Check:

- Did I solve only the requested task?
- Is this the smallest reasonable diff?
- Did I preserve existing architecture and naming?
- Did I avoid unnecessary abstractions?
- Did I avoid dynamic tricks and clever indirection?
- Did I avoid generated-looking comments and boilerplate?
- Would a senior maintainer approve this change?
