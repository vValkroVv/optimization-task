# Long-epoch runner integration and runtime-cost review

Reviewed 13 November 2026, before the new GPU pair. **Verdict: no material
source blocker to launch.** This is a bounded review of the native epoch-size
override, completed-microbatch counter forwarding and boundary receipts. It
reuses [V2 integration review](integration-review.md),
[runner checks](runner-verification.json) and [loader gate](loader-gate.json).
It does not establish a new TPS result or production acceptance.

## Contract and findings

- `run.py` accepts a positive optional `--epoch-size`, passes
  `--dataloader.dataset.epoch_size` through the existing native CLI, checks the
  requested value in resolved YAML, and records it in metadata. Unset remains
  omitted/null. Existing `StreamingTextDataLoader` passes that typed field to
  `StreamingTextDataset`, which inherits Mosaic's native resampling behavior.
  This repeats the existing fixture logically; corpus size/diversity is unchanged.
- Installed Mosaic `streaming/base/dataset.py` derives `length` as
  `ceil(epoch_size / num_ranks)` and returns it from `__len__`. At the agreed
  eight data ranks, 51,200 gives 6,400 samples/rank, 3,200 B2 batches and 100
  updates of 32 microbatches. Sixty updates consume only 1,920 local batches.
- The consumer owns raw epoch entry in both default and prefetched paths.
  `begin_prefetch_epoch` primes one epoch on the consumer; the producer receives
  only that finite iterator's remainder. Mosaic increments shared `next_epoch`
  at epoch entry. Require populated real receipts: constructor 0 (or 1 if
  another rank already entered), before-window 1, after-train 1. Combined with
  effective length and fresh start, these establish no second entry during
  updates 1–60. Null/missing fields are unavailable evidence, not a pass.
- Before-window V2 evidence must show an actual iterator, capacity 32,
  positive ready count and live named producer. The queue snapshot holds the
  existing Condition only while copying scalar metadata; it does not pause
  the producer, call `state_dict`, add barriers or mutate loader state.
- `--capture-microbatches` forwards the existing optional counter wrapper.
  Require all eight rank receipts to record exactly 32 completed actual
  `forward_backward_step` calls for every step 1–60, successful completion and
  exactly 40 measured updates. The expected global physical input count is
  `40 * 512 * 4096 = 83,886,080`; valid-target count remains a separate metric.
- Model, loss, optimizer, EMA, transfers, normalization, dtype, SAC, compile,
  parallelism and LR schedule remain on their existing paths. No training-source
  change is part of this extension. Separate bounded traces retain their prior
  collection/coverage contract and cannot supply benchmark TPS.

## Runtime costs and timing

The new epoch override changes native sample indexing/resampling and reduces
epoch-entry frequency; it adds no physical dataset duplication. The inherited
V2 queue costs remain those in the original review. Enabling the counter in
the existing completed-call wrapper adds one integer dictionary increment per
microbatch on both TPS variants (1,920 increments/rank over 60 updates); the
wrapper call and conditional were already present.

Epoch receipts perform constant scalar shared-memory reads three times per
rank. The one queue-lock snapshot and dataset reads occur before the existing
window clocks; the final epoch read occurs after native `Trainer.train()`
returns. Existing start/end CUDA drains and timed data loading/transfers remain
unchanged. There are no added device-value reads, GPU kernels, H2D copies,
collectives or compile boundaries. Boundary host work can give a producer a
small extra filling interval; both variants retain matched sampling. These
source-derived costs do not establish a speedup or a noise floor.

## Checks and limitations

The [runner verification](epoch-runner-verification.json) records 29 targeted
runner tests passed in the native `/opt` runtime, including actual typed CLI/config resolution, default omission,
requested-value mismatch rejection, schedule acceptance and metadata forwarding;
Ruff and diff checks passed. New epoch/queue receipt and timer-order tests pass.
No broad suite was rerun by this reviewer.

The wider measurement run is retained as **49 passed, 5 failed**, in
[the final log](artifacts/raw/epoch-receipts-pytest-final-20261113.log).
The [measurement verification](epoch-receipts-verification.json) retains exact
commands and corrected attempts. All five failures stop in the unchanged
cache-route layout guard before native
training or epoch receipts: nested pytest directories under this experiment's
raw root have seven relative path components, whereas supported custom-output
runs require five (`experiment/artifacts/raw/run/measurement`). Two intended
cache-error assertions therefore see the earlier layout error. This is a
harness limitation, not a passing suite or an epoch regression. Actual runner
output uses the supported shape; the prior runner receipt retains its passing
custom-output-root check. No production guard relaxation is justified by these
failures. Earlier failed harness attempts remain retained by the coordinator.

Remaining execution gates are populated all-rank epoch/queue/count receipts,
matched resolved A/B configuration, finite native loss/gradient logs, complete
TPS pair, four verified traces, and owned-process exit/GPU release. The code
review allows those authorized measurements to proceed; it does not close them.

## Reviewed identities

SHA-256:

| File | Hash |
| --- | --- |
| `runs/overnight-streaming-perf/run.py` | `2e55a19740f9244c7ad515f31c53c1be2a8f621aa36e7a71c42c51b82d99992f` |
| `runs/overnight-streaming-perf/measure.py` | `8288698d1495cfdfed66953e22b0e84fc8b93f4720702f979309d38e9be99194` |
| `tests/test_epoch_size.py` (same runner directory) | `74e7c64c02564e0f272c71cdc9de18187ee3130c177a59c42521218253e83931` |
| `tests/test_epoch_receipts.py` (same runner directory) | `c524f75d9e473fc717db636988223924ced925190fd33afa52d1010546113786` |
| Installed Mosaic `/opt/torchtitan-venv/lib/python3.12/site-packages/streaming/base/dataset.py` | `3bac8df314d3a9faac01b9f65e8a4de01420639feb67949194a6d6cbc3df02fb` |
