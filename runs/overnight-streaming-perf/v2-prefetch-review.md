# V2 independent integration and runtime-cost review

Historical source/CPU review: final preserved candidate is
`21c1891dda531a1222e6c350802f2765758794f9` on `vkropotin-v2-prefetch`, default 0.
Actual eight-owner Gloo/DCP and native/performance gates were not executed or
completed; campaign continuation is cancelled. This review preserves findings,
not runtime acceptance. The [revised protocol](../../planning/validation.md)
replaces the old native full-model/monitoring/long-window gates. No runs now.

Reviewed the uncommitted default-off candidate on `vkropotin-v2-prefetch`, base
`6123a24ff74df7f47efad677070d47ed5a238705`, under
`artifacts/raw/source-variants/v2-prefetch/gift/`. This is one source/CPU review,
against the main [V2 specification](reference-coverage.md#v2-lifetime-and-state-findings).
No GPU, distributed loader run, installation, source edit or commit was performed.

Finite acceptance: review enabled/disabled integration, cursor/RNG/epoch and error
lifetime contracts; reconcile recorded CPU evidence and the registry deselection;
identify runtime costs and leave the agreed distributed/native gates open.

## Resolved finding

**P2 — earlier cleanup errors skip producer shutdown (resolved).** The reviewed
pre-fix `gift/torchtitan/trainer.py:1017–1025` called the new `dataloader.close()` only after
callback, checkpointer and metrics cleanup. If any earlier close raises, the
owned V2 producer and persistent workers do not receive their explicit shutdown.
`CallbackManager.close()` delegates directly to `CallbackHandler.close()`, whose
loop invokes callbacks without handling exceptions (`callbacks/manager.py:129`,
`callbacks/base.py:712`). `torchtitan/train.py:76–81` also does not retry cleanup.
Thus this affects the requested failure cleanup contract, rather than an unrelated
style preference.

Pre-fix targeted CPU reproduction, from candidate `gift/`, using
`PYTHONPATH=. /opt/torchtitan-venv/bin/python3.12`:

```python
from types import SimpleNamespace
from unittest.mock import Mock
from torchtitan.trainer import Trainer
t = object.__new__(Trainer)
t.callbacks = SimpleNamespace(close=Mock(side_effect=RuntimeError("callback close failed")))
t.dataloader = SimpleNamespace(close=Mock())
try:
    t.close()
except RuntimeError:
    pass
assert t.dataloader.close.call_count == 0
```

Observed `RuntimeError callback close failed` and `dataloader.close calls: 0`.
The corrected `Trainer.close()` encloses existing cleanup in `try` and invokes
`dataloader.close()` in `finally` (`trainer.py:1017–1028`). Loader shutdown now runs
when callback, checkpointer or metrics cleanup raises. The original exception
propagates if loader shutdown succeeds; if loader shutdown also fails, its error
propagates with the prior exception retained as `__context__`. The loader's
join-timeout ownership semantics are unchanged.

Follow-up verification was limited to this finding and its affected behavior.
The reviewer ran the new
`test_trainer_close_stops_loader_when_other_cleanup_fails` (three parameter cases)
and `test_trainer_close_propagates_loader_timeout_after_cleanup_error` with
`PYTHONPATH=. /opt/torchtitan-venv/bin/python3.12 -m pytest -q` from candidate
`gift/`: **4 passed in 4.38s**. Assertions cover one loader-close call, original
exception identity and chained join-error context. This resolves the sole
source/CPU handoff blocker; no broad review or suite restart was performed.

## Integration conclusions

- Default zero preserves the original iteration/state branches. No queue, thread,
  per-batch snapshot or additional collective is created. New base lifecycle
  methods are no-ops; final streaming close explicitly releases workers.
- Enabled finite epochs construct and prime the raw Torch iterator on the
  consumer (`streaming/core/dataloader.py:66–99,215–267`). The producer consumes
  only its remainder, inherits Streaming ContextVars with `copy_context()`, and
  cannot enter a new epoch. FIFO delivery commits scalar state before yield.
  Pause surfaces stored producer errors; restore closes/joins before state
  mutation and invalidates the old generation. Timeout preserves ownership.
- Scalar `dp_rank_0` state and Mosaic replication arithmetic remain intact.
  Older same-instance restore uses public `next_epoch` and the existing explicit
  DataWorld group for two consumer-only load barriers (`:352–387`). Actual
  DataWorld construction and flattened Gloo data mesh match the specification;
  replication wrappers forward state/close only through their owner.
- Inspected unchanged text decoding, collation, FIM, Trainer accumulation/callback
  order, device transfer and checkpoint wrappers. FIM uses local
  `np.random.default_rng((seed, epoch, idx))`; no global RNG swap was introduced.
  Model/loss/optimizer/EMA/compilation paths are unchanged. Validation now closes
  its owned iteration in `finally`, retaining reusable workers.
- Installed Mosaic dataloader/dataset and Torch dataloader SHA-256 values match
  the three source pins recorded in the specification. Mosaic `_resume`,
  `_resume_incr_epoch`, `state_dict` and `load_state_dict`, and Torch worker
  cleanup were inspected. DCP retains the loader Stateful object through the
  existing `load_gfsdp_dcp` target; there is no new GPU state/checkpoint path.
  Donor prefetch mechanics and retained repository BSD license were inspected.

## Registry deselection resolved

The deselected `test_component_types_register_torch_free_and_are_pinned` fails
independently: running only that test produced one failure, with exactly the
extra `campaign_numerical_probe` callback in its registry. The test itself launches
a fresh blocked interpreter through `tests/unit_tests/torch_free.py:run_blocked`;
it cannot inherit imports from the new V2 tests. HEAD already contains the
registration in `config/schema/experiments/cuda_graph_campaign/numerical_probe.py:8`
and the stale expected list in `test_config_torch_free.py:59–79`. Those files and
`config/registry.py` have no candidate diff. `families()` intentionally imports
all schema modules. This is a pre-existing registry fixture mismatch, not V2
import pollution. Its failure remains recorded; the full suite is not a pass.

## Costs, evidence and remaining gates

The enabled path adds one producer per rank, Condition queue push/pop and scalar
Mosaic state snapshots per batch, including shared-memory reads/JSON decoding and
Python/GIL work. It starts/joins once per finite epoch and synchronously primes
the first batch. Close/pause may wait for source I/O; load adds two Gloo barriers.
There are no additional model kernels, numerical operations, H2D copies,
per-batch collectives or compile boundaries.

Depth32 holds at most32 ready batches:5MiB/rank at160KiB/batch. This is not a total
RSS or device-memory bound: worker IPC queues, active read/current consumer and
generator-frame references coexist. In particular `first` remains in the outer
generator for the epoch; its input dictionary can retain tensors after the
consumer replaces entries during device transfer. The Trainer also retains32
microbatches before compute; inherited PP wrappers perform their existing device
packing/broadcasts on the consumer. Actual retained storage, CPU contention and
integrated latency require the agreed measurements; no speedup is established.

Reused [implementation evidence](artifacts/raw/v2-prefetch-implementation-evidence/verification.txt):
306 focused CPU tests passed with one registry deselection, then five targeted
cleanup/error tests passed. After the fix, the owner recorded nine targeted
cleanup/error passes and reran scoped Ruff and whitespace checks successfully.
The 306-test suite was not repeated. This review executed the isolated registry
test and pre-fix reproduction above, followed only by the four targeted fix
cases described in the resolved finding.

**Verdict:** source/CPU handoff accepted after the cleanup correction and targeted
verification; no material source finding remains open. Actual eight-owner real-MDS
Gloo/DCP save/restore, skewed wrap and both worker persistence modes were not
executed/completed. Native/performance acceptance is incomplete; no speedup is
established and no candidate is promoted. Historical source/CPU evidence above is
preserved. Future separately authorized comparisons follow short monitoring-off
performance first and then local affected-block assessment; no native full-model
numerical, monitoring-step or mandatory new-trace gate remains. No full-model
checkpoint or additional parallelism matrix is required by this loader-only scope.
