# Source import verification

Recorded: 2026-10-27. This receipt describes a local source snapshot import, not a training run.

## Source selection

| Snapshot | Remote main commit |
| --- | --- |
| `gift` | `f840cf25e4294f93a5c91a20707c30b97e3c1dbf` |
| `llm-foundry` | `1ec1360304a182844372f6c3989c3f10288fcba8` |

Remote `refs/heads/main` was fetched into isolated temporary clones. The original root checkouts were read-only. Recursive submodules use each recorded gitlink pin; their branch names do not override those pins. Credential-free source URLs, all 50 component pins/trees, and every imported file hash/mode appear in [source-baseline.json](source-baseline.json).

## Results

- **PASS:** 56,049 imported paths, including symlinks, match the committed upstream paths.
- **PASS:** Every imported file SHA-256, byte length, executable mode, and symlink target was checked after export.
- **PASS with explicit exceptions:** All original source bytes match upstream except the three credential literal substitutions below.
- **PASS:** The three sanitized scripts parse with `ast.parse`; `ruff check --select F821,F811,F841 --no-cache` on those files exits 0.
- **PASS:** 48 recursive submodule snapshots are ordinary files; no nested `.git` files/directories or gitlinks were copied.
- **PASS:** No original working-tree dirty/untracked content or Git administrative metadata was copied.
- **PASS:** Both original checkout HEADs, branches, statuses, and binary diff digests remained unchanged. The pre-existing GIFT submodule modification remains present.
- **PASS:** No committed individual blob reaches 50 MiB; no Git LFS substitution was performed.

## Credential publication exceptions

A tracked `GITLAB_TOKEN` literal was replaced with `REPLACE_WITH_GITLAB_TOKEN` at line 30 in each imported file below. The rest of each file is unchanged. This intentionally breaks byte parity for these files and requires a safe credential setup before those auxiliary submission scripts can authenticate. The upstream original checkouts and temporary source objects were not altered. No token value or token-specific hash is recorded.

- `gift/contrib/pretrain_auto_metrics/contrib/gigachat_utils/convertation/submit_tests.py`
- `llm-foundry/contrib/gigachat-utils/convertation/submit_tests.py`
- `llm-foundry/contrib/pretrain_auto_metrics/contrib/gigachat_utils/convertation/submit_tests.py`

The manifest records each upstream blob SHA and sanitized full-file SHA-256. No source file was excluded. Other scan matches were variable/template credential URLs, synthetic test/data URLs, assertions for PEM headers in generated test keys, and the canonical AWS documentation example. The scoped scan does not certify the absence of every possible upstream secret.

## Single-repository layout and CI

The snapshots deliberately retain `.gitmodules` for provenance. They are not active task-repository submodules: run task Git commands at the repository root, and do not run `git submodule update` to refresh these copies. Updating a baseline requires another explicit pinned import. Nested source workflows and `.gitlab-ci.yml` files are preserved verbatim; they are not root GitHub workflows and do not automatically run on publication.

## Runtime and cost review

This import adds no training implementation, configuration, tensor allocation, host/device synchronization, communication, compilation, or repeated runtime work. No training, dependency installation, unit tests, GPU jobs, or performance benchmarks were run. Source integrity checks establish snapshot provenance only; they do not establish runtime compatibility or production training correctness. The three credential-only changes affect authentication in auxiliary conversion submission scripts.

## Reproduction scope

The import enumerated each committed tree with `git ls-tree -rz HEAD`, fetched its exact recursive gitlink revisions, streamed blob bytes with `git cat-file --batch`, and exported ordinary files/symlinks with committed executable modes. This avoids archive filters and working-tree smudge/checkout transformations. A second filesystem pass checked all paths, byte lengths, hashes, modes and symlink targets against the export record. Use the manifest file list when staging: inherited nested `.gitignore` rules can otherwise hide committed upstream snapshot paths.
