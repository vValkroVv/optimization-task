---
name: pam-code-style
description: >
  Forces the minimal solution that actually works and matches the repository style. Apply YAGNI, reuse existing code before writing new code, avoid new dependencies, and keep the diff as small as correctness allows.
---

Work as a careful maintainer of this repository. Implement requested changes with the smallest reasonable diff. Preserve the existing architecture, style, naming, and public behavior.

## Persistence

ACTIVE EVERY RESPONSE. No revert after many turns. No filler drift. Still active if unsure.

## The ladder

Before writing any code, stop at the first rung that holds:

1. **Does this need to exist at all?** Speculative need = skip it, say so in one line. (YAGNI)
2. **Already in this codebase?** A class, method, domain object, or pattern that already lives here → reuse it. Look before you write; extend the existing concept instead of creating a parallel one.
3. **Already-installed dependency solves it?** Use it. Never add a new one for what a few lines can do.
4. **Does this abstraction pay for itself?** Add abstractions only when they reduce real duplication or complexity. Keep business logic in the existing class/domain object; do not move it into generic
helpers or procedural pipelines.
5. **Only then:** the minimum code that works.

The ladder runs after you understand the problem, not instead of it: read the task and the code it touches, trace the real flow end to end.

**Bug fix = root cause, not symptom.** For bugs, grep callers and fix the shared root cause, not one visible symptom.

## Rules

### Avoid dynamic and clever code
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

### Forbidden by default
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
  - defensive guards with no named failing case (model-name slugs, `isinstance` checks on results, normalization on top of a pydantic model) — if you cannot name the input that breaks without the guard, do not add it
  - submodule pointer changes that are not the point of the change
  - a constant introduced for a single use that carries no domain meaning
  - fields and computed values with no downstream consumer
  - hardcoded environment-specific values (endpoints, hosts, paths)

### Refactoring

Refactor only when it directly supports the requested change.
Do not mix large refactors with feature work.
If a useful refactor is not necessary for the task, mention it separately instead of doing it.

### Blast radius

A feature change must not quietly alter shared core or the behavior of neighboring components.
Touching shared code or another team's component: agree it up front, or split it into its own change.
Do not move code between files without a stated reason.

### Comments

Do not add obvious comments.
Code should explain itself through names, structure, and types.
Comments are acceptable only for non-obvious business rules, external constraints, historical reasons, or surprising behavior.
A docstring is warranted only where the transformation is non-obvious, and then it states what comes in, what goes out, and which cases it covers — not a retelling of the code.
Do not leave TODOs unless explicitly requested. Work deferred on purpose is recorded as a follow-up with an explicit scope and owner, in the change description or an issue — not as a vague "later" in the code.

### Docker and data

Benchmark data and vendored tools belong in the image, not on NFS and not in `contrib/`.
Clone repositories in `gitlab-ci.yaml` and `COPY` them into the image; do not clone inside the Dockerfile.
Do not run tests inside a Dockerfile.
Duplicated layers go into their own build stage instead of being copy-pasted into the final image.
Pass secrets through docker secrets, never build args or files in the image.
Pin dependencies and keep the source reachable; an unpinned requirements file is a future breakage with no owner.

### Tests

Do not add or update tests unless the user explicitly asks for tests.
One exception: a fix to caching, resumption, locking or restart logic ships with a regression test reproducing the original breakage — those bugs are invisible on a green run and only surface when a job is preempted at the wrong moment.
When the user asks for tests, follow `/pam-tests`: `.agents/skills/pam-tests/SKILL.md`.
When changing behavior without adding tests, identify relevant existing tests and mention what should be tested in the final note.

### Running checks

Do not run broad or expensive test suites unless explicitly requested.
You may run a narrowly scoped existing test, formatter, type checker, or linter when it directly validates the changed code and does not require expensive environment setup.
If no checks are run, state that clearly in the final note.

## Before finishing

Check:
  - Did I solve only the requested task?
  - Is this the smallest reasonable diff?
  - Did I preserve existing architecture and naming?
  - Did I avoid unnecessary abstractions?
  - Did I avoid dynamic tricks and clever indirection?
  - Did I avoid generated-looking comments and boilerplate?
  - Did I avoid adding tests unless explicitly requested, and avoid running broad or expensive suites unless requested?
  - Would a senior maintainer approve this change?

## Output

Final note:
  - What changed.
  - Why this is the minimal reasonable solution.
  - Any trade-offs or limitations.

Code first. No essays, no feature tours, no design notes. Explanation the user explicitly asked for (a report, a walkthrough, per-phase notes) is not debt, give it in full, the rule is only against unrequested prose.