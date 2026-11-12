# Public CUDA Graph step boundary implementation

## Scope and source state

Target: task-local `gift/`, based on repository HEAD `446431dfa191ef0300f116257ae918bc85e38c06`; implementation follows the preserved `SOURCE-SNAPSHOT/002` contract. Explicit source-freeze release received from the owner after baseline attempt 001 aborted with exit 1 and baseline 002 was not run. No GPU query, CUDA initialization, GPU execution, dependency installation, package mutation, or publication is authorized or performed here.

The selected callback marks one public CUDA Graph Trees generation per complete optimizer update. It must remain optional by omission, use native `Trainer.train_step` callback events, preserve four accumulation microbatches and the existing optimizer, and leave CampaignTrainer/RNG continuation, the scan correction, model/loss code, compile options, FSDP2, EP8, selective AC, FP8 and attention policy unchanged. Strict independent-run A/A loss and norm differences remain diagnostic, not eligibility gates; this change does not attempt to eliminate other baseline nondeterminism.

## Finite acceptance checklist

- Add the torch-free callback schema and runtime Observer in the campaign experiment package; register and pin its exact type.
- Reject PP at `fit_start`, reject a second instance with `validate_dependencies(previous)`, and call the public marker exactly once at each native `batch_start`.
- Keep marker absent by default; document its runtime-control purpose and target limitations.
- Verify actual callback-manager event ordering and a lightweight real native `Trainer.train_step` with four mocked forward/backward microbatches; verify configuration decode, duplicate and PP rejection, exception propagation, no RNG-state changes, no CUDA use, and no effect on existing continuation requirements.
- Run only focused CPU tests, changed-file F821/F811/F841 lint, formatting/whitespace checks and a final scoped review. Record failures without dropping gates.

## Pre-code runtime-cost review

- Computation/time: one existing callback-dispatch invocation and one call to PyTorch's compiler boundary per optimizer update. No added tensor pass, kernel, copy, device-value read, synchronization or collective is expected. Frequency is once per full update, independent of accumulation count.
- Memory: callback keeps no tensor or checkpoint state. A longer graph generation can retain more graph-pool/recording state; peak CUDA memory and capture/recapture behavior remain a full-target GPU gate.
- Host/device: generation bookkeeping is host-side. No device synchronization, transfer, allocation or scalar extraction is designed. When the existing callback handler already builds an observer snapshot, the callback reuses that dispatch snapshot; enabling any Observer otherwise uses the handler's normal snapshot path.
- Communication: none added; all EP/FSDP communication and ordering remain unchanged.
- Numerical/lifecycle: the callback changes compiler iteration ownership, so CPU event-order checks cannot establish full capture safety, numerical equivalence, checkpoint continuation or throughput. Those require the separately authorized target gates and will not be claimed from this implementation.

## Source evidence reviewed before implementation

`Trainer.train_step` emits `Event.batch_start` before zero-grad, data fetch, mask construction, all microbatch forward/backward work, clipping and optimizer update (`gift/torchtitan/trainer.py:719-807`). `CallbackManager` builds configs in mapping order (`gift/torchtitan/callbacks/manager.py:39-57`), and `CallbackHandler` invokes observers in that order and propagates exceptions (`gift/torchtitan/callbacks/base.py:513-571,630-667`). Schema modules are discovered through `pkgutil.walk_packages` (`gift/torchtitan/config/registry.py:19-25`); callback type names are pinned in `gift/tests/unit_tests/test_config_torch_free.py`. The timing observer's current boundary behavior and configuration are in `gift/torchtitan/experiments/cuda_graph_campaign/observer.py` and its schema.

## Implementation and verification

Added:

- `gift/torchtitan/config/schema/experiments/cuda_graph_campaign/step_boundary.py`: torch-free `campaign_cudagraph_step_boundary` schema with a lazy owner spec and duplicate-type rejection through callback dependency validation.
- `gift/torchtitan/experiments/cuda_graph_campaign/step_boundary.py`: optional Observer; rejects PP in `fit_start` and makes exactly one public marker call in `batch_start`.
- `gift/tests/unit_tests/callbacks/test_campaign_cudagraph_step_boundary.py`: config round-trip and default omission, native config without RNG continuation, unchanged continuation config with the marker, duplicate and PP rejection, event order through the actual callback manager/native `Trainer.train_step` with four mocked compute microbatches, Python/NumPy/Torch CPU RNG preservation, and marker-exception propagation.
- `gift/tests/unit_tests/test_config_torch_free.py`: added the callback type to the existing exact sorted registry pin; prior campaign callback entries in that dirty shared file were preserved.
- `gift/torchtitan/experiments/cuda_graph_campaign/README.md`: callback config, ordering, PP limit, runtime-control cost and separate campaign gates; bitwise independent-run A/A differences are diagnostic only.

The run overlay remains owner-managed; this implementation does not edit it. No core trainer, continuation schema/runtime, scan fix, model, loss, compile preset, graph option, FP8, attention or parallelism files were changed by this subtask.

Commands and results, run from `gift/` with `/opt/torchtitan-venv/bin/python` (Python 3.12.3), pytest 7.3.2, ruff 0.14.10:

- `pytest -q tests/unit_tests/callbacks/test_campaign_cudagraph_step_boundary.py tests/unit_tests/test_config_torch_free.py` — **123 passed**, 3 deprecation/future warnings.
- After adding the full duplicate validation and CPU RNG checks and final doc wording, `pytest -q tests/unit_tests/callbacks/test_campaign_cudagraph_step_boundary.py tests/unit_tests/test_campaign_continuation.py` — **25 passed**, 2 warnings.
- `ruff check --select F821,F811,F841 --no-cache torchtitan/config/schema/experiments/cuda_graph_campaign/step_boundary.py torchtitan/experiments/cuda_graph_campaign/step_boundary.py tests/unit_tests/callbacks/test_campaign_cudagraph_step_boundary.py tests/unit_tests/test_config_torch_free.py` — **passed**.
- `ruff format --check` on the same four Python files — **passed**. The initial format check identified the new test file; `ruff format` corrected it before final checks.
- `git diff --check -- tests/unit_tests/test_config_torch_free.py torchtitan/experiments/cuda_graph_campaign/README.md` — **passed**. A trailing-whitespace scan of the new runtime, schema and test file returned no matches. No local Markdown links were added.

The focused CPU integration test patches the public compiler marker and runs the actual inherited `Trainer.train_step`, so it proves callback timing/order and exception propagation without executing CUDA. It verifies two updates, one mark before each update's zero-grad/fetch, all four mocked forward/backward boundaries, then clipping and optimizer step. Python, NumPy and Torch CPU RNG states are unchanged. No CUDA RNG query, CUDA initialization, GPU query, GPU run, install or external write was performed. The `torch.cuda` import warning emitted by the test runner is environmental; no CUDA API was called by this change/test.

## Review and remaining gates

Source review confirms the schema does not import Torch, schema discovery is automatic, the callback has no enable flag, trainer registry entry, dependency or checkpoint state, and the only compiler API call in the runtime callback is at `batch_start`. The event is emitted first by native `Trainer.train_step`; the test manager builds timing before the marker and observes that order. Existing campaign continuation tests pass with the callback implementation present.

Source implementation and focused CPU checks are complete. Independent source/integration review is pending with the owner. Owner-managed overlay ordering is pending. Actual-target graph-on capture and replay, changing-mask lifetime, matched numerical/FP8/checkpoint continuation, paired graph-off/on throughput and memory, and sustained execution remain open GPU gates; no production behavior or speedup is claimed. Strict independent-run loss/norm bitwise A/A results remain diagnostic only under the owner's latest acceptance correction.

SHA-256 at handoff:

- `gift/torchtitan/config/schema/experiments/cuda_graph_campaign/step_boundary.py`: `1fa6cb155a62b1afd55468c1317782c18b2a3a37be9ed7315ec1302bff870c68`
- `gift/torchtitan/experiments/cuda_graph_campaign/step_boundary.py`: `a6bfba5cdaa2fc1e29a553d47c41004391918637e9cf723e90b5b60e9b4b098b`
- `gift/tests/unit_tests/callbacks/test_campaign_cudagraph_step_boundary.py`: `13d8f603061528ee356a058b16dcb2135eb6eeaae527ee6bf98675e18037a0bd`
- `gift/tests/unit_tests/test_config_torch_free.py`: `903c140ae14d7d161a536a2c5ed10d751930a2533635176cef8073b046c269af`
- `gift/torchtitan/experiments/cuda_graph_campaign/README.md`: `1f6911dcb28d42e41ec7277e1e282e8056221173e7637ef5fea1294ec8b84869`

## Verification qualification added after independent review

The first broad command above included the entire pre-existing `tests/unit_tests/test_config_torch_free.py` module. Independent review found that its existing `test_flex_flash_hardware_check_is_container_only` calls `has_cuda_capability`, which can query or initialize CUDA on this host. The recorded run result remains **123 passed**; however, the test command was not strictly CPU-only, and the exact hardware-query/initialization state for that run was not recorded. The earlier blanket statements that no CUDA API/query/initialization occurred during that broad test run are therefore withdrawn. This does not mean a training or benchmark run occurred; neither was intentionally launched by this subtask.

The final marker plus continuation command remains **25 passed** and is the focused behavior evidence for this change. Independent review also reports a separately guarded **26-case CPU gate** as pertinent. These checks do not establish GPU capture, replay, memory, or performance. No source, test, or GPU rerun was made for this evidence correction.
