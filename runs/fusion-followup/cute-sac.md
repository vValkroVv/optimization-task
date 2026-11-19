# CuTe gated norm statistics during SAC replay

Native rank-zero backward graphs confirm 33 CuTe calls per three microbatches
whose BF16 activation `y` is dead. The statistics-only variant preserved every
output and gradient bit in the two captured norm components on H100. Complete
compiled forward and backward CUDA Graph replay fell from 364.802 to 303.931 µs
for the plain norm and from 409.370 to 331.060 µs for the residual norm. Both
allocator peaks stayed flat in each component. [Replay results](results/cute-sac.json)

The owner selected this replay for the canonical production launch after the
native candidate run. See the [native follow-up report](report.md) for its
pre-promotion result, provenance and measurement limits.

## Native graph uses

The [liveness analyzer](cute_liveness.py) matches each profiled
`CompiledFunctionBackward` scope by the ordered CuTe calls and correlated
Inductor Triton launches. It then inspects executable tuple reads in the
generated wrapper. It recognizes a dead activation only when no read selects
index zero and no consumer reads the whole tuple. The cache contains multiple
rank-specific or equivalent wrappers with the same signature. Each matching
wrapper agreed on the liveness of each call in both traces.

| Replay operation | Hidden width | Calls with dead `y` | Calls with live `y` |
| --- | ---: | ---: | ---: |
| Residual CuTe norm | 4096 | 24 | 30 |
| Plain CuTe norm | 4096 | 9 | 54 |
| Plain CuTe norm | 512 | 0 | 9 |

Each trace covers 33 rank-zero compiled backward scopes containing CuTe norm
replay across three microbatches. The newer production control has 2.056 ms of
residual and 0.747 ms of plain output-stage kernel time on the dead-output
calls. These durations sum device kernels within the profile window. They do
not measure elapsed step time or the exposed cost in the native schedule.

The `GIFT_CUTE_NORM_SAC_STATS_ONLY=1` pass runs after AOTAutograd fixes
the forward and backward graph interfaces. It keeps the full op for a live
`y` or an opaque tuple consumer. The replacement keeps the existing weight
preparation, F1, FP32 split reduction and six inputs to the VJP. Its finalizer
does not allocate `y`, stage residual data or run the gate/output epilogue.
The original `B2/S4096/H4096` BF16 allocation for `y` is 64 MiB at each eligible
replay. It was transient in the component and did not set the allocator peak.
The rewrite adds no saved activation or communication. The output rounding and
residual identity gradient remain in their original paths. The canonical
production launcher enables this pass by default; the run report retains the
explicit historical comparison path.
[Graph pass](../../gift/torchtitan/models/common/norms/cute_gated_norm/compiler.py) ·
[CuTe kernels](../../gift/torchtitan/models/common/norms/cute_gated_norm/forward.py) ·
[Pinned TorchInductor pass sequence](https://github.com/pytorch/pytorch/blob/8b353065ce260336c5fdfa74d754e8bd9b91b062/torch/_inductor/fx_passes/post_grad.py)

## Native-input component

The rank-zero fixture comes from update 1, microbatch 1 of
`followup-fixtures-001` at `0f0bf5604`. Both variants have the native activation,
gate parameters and output. The plain case has the tangent from the same call.
The residual case shares a same-shape native tangent from another CuTe call
between baseline and candidate. The replay uses the GIFT selective checkpoint
wrapper and the normal per-block Inductor compiler. It covers a single norm
component on GPU 1, without the surrounding transformer block or collectives.

Both original forward outputs match the captured bytes. Compiled output, input
gradient, three parameter gradients and the residual gradient where present
match bitwise. `rho`, `P` and the four prepared weights each match bitwise at
8192, 65 and zero rows. A squared-output control uses `y` in the VJP and
retains the full replay with identical output and gradients. The traces show
one ordinary forward, one statistics replay and one VJP in each candidate.
Baseline traces show two ordinary forwards and one VJP. Four existing CuTe
Hopper tests for compilation, empty rows and op registration passed, plus the
noncontiguous and unaligned-input test. Nine CPU compile-default and norm
config tests passed.

| Measurement | Plain, baseline → stats | Residual, baseline → stats |
| --- | ---: | ---: |
| Warm CUDA Graph replay, forward and backward median | 364.802 → 303.931 µs (−16.69%) | 409.370 → 331.060 µs (−19.13%) |
| Baseline round medians for CUDA Graph replay | 363.525 / 368.965 µs | 409.501 / 406.974 µs |
| Candidate round medians for CUDA Graph replay | 303.584 / 308.854 µs | 332.101 / 328.976 µs |
| Replay epilogue kernel alone, trace | 56.095 → 4.320 µs | 78.367 → 4.480 µs |
| Complete component kernel sum, trace | 366.394 → 301.656 µs | 409.211 → 324.475 µs |
| Complete component device span / gaps, trace | 1263.536 / 897.334 → 799.823 / 498.391 µs | 894.137 / 484.926 → 829.846 / 505.371 µs |
| Peak allocated / reserved, complete component and isolated backward | 271.859 / 280 MiB in both arms | 335.859 / 344 MiB in both arms |

CUDA Graph figures use pooled medians from two warm rounds per arm in
baseline–candidate–candidate–baseline order, with 25 samples per round and 20
replays per sample. Capture starts on a separate stream after five warm calls.
The replay contains the component device work and graph dispatch and excludes
per-call Python, allocator and autograd submission. Traces use an ungraphed
single-stream call and include host-fed queue gaps. Each arm has 10 kernels and
zero copy or memset events. The finalizer uses 32 registers per thread and zero
shared memory versus 168 registers and 160 KiB for the plain epilogue or
202 KiB for the residual epilogue. Baseline residual epilogue times already
include the fused residual addition.

Unprofiled calls were also timed without a CUDA Graph. Per-round medians for
plain host submission were 674 and 1074 µs in baseline and 693 and 683 µs in
candidate. Corresponding synchronized wall medians were 757 and 1100 µs versus
756 and 746 µs. The two baseline observations bracket enough drift to make the
plain host or integrated latency gain inconclusive. Residual host submission
round medians were 748 and 726 µs versus 697 and 694 µs, with synchronized
wall time 812 and 793 µs versus 757 and 751 µs. Event-record-and-wait medians
were 81 and 26 µs versus 62 and 64 µs for plain, and 65 and 67 µs versus 60 and
60 µs for residual. The component result does not establish training TPS,
training peak memory or all-rank activation.

## Reproduction and provenance

On SR008, use `7f8f40c1fbcb83334407adacd6e137bfdc52e57d`, H100 80 GB, CuTe DSL
4.4.1 and PyTorch `2.15.0.dev20260822+cu130` at `8b353065ce260`. The
source rewrite is in `c85c0d9cf` and `ca5f9972f`. The first smoke caught a stale
Inductor input-layout metadata signature. The latter revision refreshes that
metadata and the rerun passed. `cute-component-002` also passed but used
deterministic allocation debug fills outside the native setting. Use
`cute-component-003` for timing, which reads `debug.deterministic=false` from
the fixture's resolved configuration and has no synthetic fill kernels.

```bash
cd /workspace-SR008.nfs2/vkropotin/optimization-task
source /etc/profile.d/wm.sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD/gift${PYTHONPATH:+:$PYTHONPATH}" \
  /opt/torchtitan-venv/bin/python3.12 -B runs/fusion-followup/cute_component.py \
  --fixture runs/kernel-fusion/artifacts/raw/followup-fixtures-001/component-inputs/cute-rank0.pt \
  --resolved-config runs/kernel-fusion/artifacts/raw/followup-fixtures-001/native/config/resolved_config.yaml \
  --output runs/fusion-followup/artifacts/raw/cute-component-003
```

Use a fresh output directory to repeat the command. The raw artifact base is
`runs/fusion-followup/artifacts/raw/` on SR008. `cute-component-003` contains
both trace pairs, the live-output control, generated wrappers and the result
receipt. The [checked-in receipt](results/cute-sac.json) matches that run's
`result.json` byte-for-byte. The receipt records hashes of all five component
traces, the source fixture and its resolved configuration.

The two native liveness receipts below are under the same raw artifact base.
Replay the analyzer with the cache at
`runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache/torchinductor`
and `runs/kernel-fusion/artifacts/raw/<run>/native/profiling/traces/iteration_21/rank0_trace.json`
as `--trace`. Each receipt records the trace hash, matching wrapper files and
hashes, and original call lines and tuple reads.

| Native trace run | Receipt | Receipt SHA-256 |
| --- | --- | --- |
| `fusion-cute-residual-001-trace` | `cute-liveness-001.json` | `dfc10247eac06bd22329868d6933bf4b73dd82e05db455d26f259ca8499e84c5` |
| `backward-native-001-control-trace` | `cute-liveness-backward-native-001.json` | `91f4159acfb891e7cc94fe3a2b6f0aabefe788b799309a79ab3d6256f7484f70` |
