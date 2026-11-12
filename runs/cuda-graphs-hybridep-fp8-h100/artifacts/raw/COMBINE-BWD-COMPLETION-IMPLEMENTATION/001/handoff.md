# Combine backward completion diagnostic implementation

## Result

Implemented the accepted task-only diagnostic in `gift/` and froze its source
for review. It replaces only the pinned module-global
`hybridep._run_on_comm_stream` helper through the new explicit entrypoint. The
ordinary `train` and `train_scan_fix` paths remain unchanged. No core hook,
Torch private dispatcher field, package change, config preset change, or
benchmark claim was added.

The adapter calls the saved helper once with the original callable and input
objects. It selects only the pinned combine-backward lambda. After the native
helper returns during updates 21–40, it flushes `checkpoint_enter`, records a
no-timing event on the current stream, synchronizes only that event, then
flushes `checkpoint_complete`. It returns the original output object. The
observer uses native step and microbatch callback boundaries, writes one
rank-local JSONL file, and marks any active microbatch without a selected call
as invalid evidence. The entrypoint preloads and validates the callback pairing,
installs the scan correction before the adapter, runs native main, and restores
the helper in `finally` while preserving a primary error.

## Files

- `gift/torchtitan/experiments/cuda_graph_campaign/combine_bwd_completion.py`
- `gift/torchtitan/config/schema/experiments/cuda_graph_campaign/combine_bwd_completion.py`
- `gift/torchtitan/experiments/cuda_graph_campaign/train_combine_bwd_completion.py`
- `gift/tests/unit_tests/test_campaign_combine_bwd_completion.py`
- `gift/tests/unit_tests/test_config_torch_free.py` (pins the new callback type)
- `gift/torchtitan/experiments/cuda_graph_campaign/README.md`

The runtime verifies the current task HybridEP source path and SHA-256 pin
`12b1921372f265ede5cca4974f5867f954f84862cc257ff8e02bb7ecafa0814d`; the
focused test also checks that this pin matches the existing GIFT source file.

## Performance and behavior limits

The normal training entrypoints have no added import or runtime work. The
diagnostic path adds a Python selector on HybridEP helper calls. Selected calls
in updates 21–40 add one no-timing event record and blocking host wait, plus
flushed local JSONL records. This changes overlap and object lifetime and can
mask a timing race. The CPU tests do not validate CUDA, closed-extension peer
progress, NCCL, kernel correctness, performance, numerical effects, or the
originating fault. The source-supported design remains restricted to the exact
one-node eight-rank target.

## Verification

Guarded focused invocation used
`/opt/torchtitan-venv/bin/python3.12` from `gift/`, with
`CUDA_VISIBLE_DEVICES=''`. Before importing task tests it asserted
`torch.cuda.is_available() is False` and `torch.cuda.device_count() == 0`, then
replaced both `torch.cuda.init` and `torch.cuda._lazy_init` with functions that
raise. It ran:

```text
tests/unit_tests/test_campaign_combine_bwd_completion.py
tests/unit_tests/test_config_torch_free.py::test_config_package_imports_torch_free
tests/unit_tests/test_config_torch_free.py::test_component_types_register_torch_free_and_are_pinned
```

Result: **38 passed in 4.28s**. The full output is preserved in
`pytest-focused-003.log`. The first guarded invocation is also preserved in
`pytest-focused-001.log`; it had 17 passes and six test assertion failures
because the test matched `output_dir` in an error string instead of checking
`ConfigFieldError.field`. The corrected invocation before the final extension
of the tests is `pytest-focused-002.log`.

The required scoped lint passed:

```text
ruff check --select F821,F811,F841 --no-cache \
  torchtitan/experiments/cuda_graph_campaign/combine_bwd_completion.py \
  torchtitan/experiments/cuda_graph_campaign/train_combine_bwd_completion.py \
  torchtitan/config/schema/experiments/cuda_graph_campaign/combine_bwd_completion.py \
  tests/unit_tests/test_campaign_combine_bwd_completion.py \
  tests/unit_tests/test_config_torch_free.py
All checks passed!
```

`git diff --check` passed; the trailing-whitespace scan found no matches. The
campaign README has no local links. The pinned HybridEP, train, trainer,
callback, registry, scan-fix entrypoint and scan-fix backend hashes still match
`RECURRENT-CUDA-FAULT-DIAGNOSTIC-CONTRACT/001/source-pins.json`.

Independent implementation review found and verified fixes for the HybridEP
path calculation, zero-call active microbatch evidence, and log-close behavior
when cleanup fails during a primary exception. It found no remaining material
code issue. The review pins are being recorded separately by the review owner.

## Remaining gates

No GPU query, initialization, training, benchmark, or diagnostic attempt was
run. The root owner retains the one authorized graph0 capture/diagnostic
attempt and outcome classification. Do not infer target activation or peer
progress from these CPU checks.
