# AGENTS.md

## Scope

These instructions apply when the user explicitly asks you to add, update, review, or discuss tests under `sber_tests`.

Do not create tests just to increase coverage numbers. Tests should protect meaningful behavior and make future changes safer.

## Running tests

Do not run tests unless the user explicitly asks you to run them.

The test environment may be unavailable, incomplete, slow, or dependent on cluster-specific packages and paths. If you add or change tests without running them, say so clearly in the final response.

## Test structure

Follow the existing `sber_tests` layout.

Mirror the source module path from `node_runner/src` under `sber_tests/unit` when adding unit tests.

Use `pytestmark = pytest.mark.unit` for unit test modules.

Prefer existing fixtures from `sber_tests/conftest.py` and helpers from `sber_tests/test_support` before adding new shared test code.

## What to test

Test externally visible behavior, contracts, inputs, outputs, error handling, and important edge cases.

Do not test private implementation details unless there is no practical behavioral surface to test.

Avoid tests that only lock in the current internal structure, call order, temporary variables, or helper decomposition.

For bug fixes, write tests that would fail on the bug and pass because the behavior is fixed.

For config parsing and model validation, test accepted inputs, normalized outputs, defaults, aliases, and validation errors that matter to users of the config.

For integrations with SGLang, OBM, tokenizers, subprocesses, filesystems, or networks, mock the boundary and assert the behavior at this repository's boundary.

## Test design

Use parametrized tests for related input/output cases instead of copy-pasting nearly identical tests.

Keep each test focused on one behavior. If a test needs many assertions, make sure they describe one coherent outcome.

Use clear arrange-act-assert structure. The expected value and the actual value should be easy to see in the test body.

Prefer explicit expected data over clever generated expectations.

Keep test names specific enough to explain the scenario and expected behavior.

## Fixtures and helpers

Move repeated setup, mocks, fake classes, and reusable builders into fixtures or small reusable helpers.

Keep fixtures boring and transparent. Avoid fixtures that hide the important input, expected output, or assertion logic.

Do not bury the core comparison inside several helper layers. A maintainer should understand what is being compared without jumping through the whole test support stack.

Use fake classes when they make a dependency boundary obvious. Use mocks when the exact behavior can be described simply.

Avoid over-general test utilities that become a second implementation of the production code.

## Assertions

Assert behavior directly and readably.

Prefer comparing complete simple values, dictionaries, lists, statuses, payloads, and raised errors when that makes the behavior clear.

Use partial assertions only when full equality would make the test brittle or obscure the intent.

Do not assert incidental details such as unrelated default fields, object identity, logging noise, or exact call order unless those details are the behavior being tested.
