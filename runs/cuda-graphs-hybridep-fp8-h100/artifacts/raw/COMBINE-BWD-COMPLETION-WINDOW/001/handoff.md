# Combine backward completion window 21–60

## Result

Extended the opt-in completion diagnostic through update 60. Its fixed native
stop is now 60, validated against `_LAST_UPDATE`; the rejection message names
`training.steps=60`. The focused runtime test exercises fit-start acceptance at
60 and rejection at 40 and 61. The boundary test observes real adapter calls
inside the window at 48 and 60, and confirms calls at 20 and 61 do not wait on
CUDA events. Full fixture coverage now expects 160 selected calls and completed
checkpoints (one per active microbatch, not a GPU call-count prediction).

The maintained diagnostic README now says updates 21 through 60. Callback
schema, entrypoint, selector, HybridEP pin, first-20 inactive behavior, and
production preset are unchanged. The pre-edit runtime-cost analysis is in
`performance-review.md`.

## Scope and pins

Task checkout: branch `main`, base `446431dfa191ef0300f116257ae918bc85e38c06`.
The three source/test/README targets were already untracked task-local files at
the start of this subtask; no commit or source-baseline write was made. Final
SHA-256 pins:

- `gift/torchtitan/experiments/cuda_graph_campaign/combine_bwd_completion.py`:
  `b07e53b3f6c40e89bbdd8e243f8ec061a29e8dd29574691ea375c96274c0ab6a`
- `gift/tests/unit_tests/test_campaign_combine_bwd_completion.py`:
  `012cfac1075eed82b8171a9482376fbf990b5f1eefeeba7f225fefc322a23a6f`
- `gift/torchtitan/experiments/cuda_graph_campaign/README.md`:
  `04e23c6c92eb0d59d8ffa0c871fee4e319815200cabcbf48e0bcfdf802676751`

## Checks

- `python -m pytest -q tests/unit_tests/test_campaign_combine_bwd_completion.py`
  from `gift/`: **39 passed**, one upstream `pynvml` deprecation warning. Full
  output: `focused-pytest.log`.
- `ruff check --select F821,F811,F841 --no-cache tests/unit_tests/test_campaign_combine_bwd_completion.py torchtitan/experiments/cuda_graph_campaign/combine_bwd_completion.py`
  from `gift/`: **All checks passed**. Output: `ruff.log`.
- The native-stop test was run against a temporary `_LAST_UPDATE = 40`. It
  failed all three parameters as expected; the original source bytes were
  restored in `finally` and compared byte-for-byte. Output:
  `regression-sensitivity.log`.
- The README check found no Markdown links and no trailing whitespace. The
  scoped `git diff --check` command passed; because the task-local target files
  are untracked, each also received `git diff --no-index --check /dev/null
  <path>`, with no whitespace diagnostics. Output:
  `scoped-whitespace-check.log`.
- A first regression-check shell wrapper had a heredoc syntax error before
  executing Python. A first whitespace-log destination was relative to the
  wrong working directory. Neither changed source; the corrected checks above
  completed and their outputs are recorded.

No GPU query, GPU run, package installation, benchmark, or shared-resource
action occurred. GPU activation/progress and the conditional 60-update probe
remain open gates. Source006 was left untouched; the coordinator can review
these files and create the next integrated snapshot.
