# Capacity destination-group all-reduce

## Scope and finite acceptance

Authorized by the 15 November 2026 goal attachment. Starting task revision is
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`; the initial working tree was clean.
Change only the aggregation order in `select_capacity_routes`: retain the local
expert histogram, sum it into destination-group loads before the functional EP
SUM, and preserve the existing capacity predicate and selection.

At E128/EP8 the collective tensor changes from 128 to 8 int64 elements,
1024 to 64 payload bytes. This is a tensor-payload statement, not wire traffic
or a latency claim. The collective count, host predicate and SAC policy remain
part of the requested contract. Moving the group sum may change compiler fusion,
launches and scheduling; component and integrated measurements must establish
their actual effect. No persistent state or configuration switch is introduced.

- [x] Integrated minimal patch and reproducible original-order control.
- [x] Focused exact behavior, payload regression, gradient, Gloo, compile/SAC
  checks and scoped lint, with failures/skips retained.
- [x] One scoped independent integration/runtime-cost review; findings resolved.
- [x] Live identity/ownership and all eight H100s verified on the existing
  owner allocation, without new allocation or environment changes.
- [x] One warmed alternating eight-rank functional NCCL measurement of both
  the all-reduce and full aggregation fragment, with group-load parity,
  per-rank distributions and slowest-rank results.
- [ ] Exactly one matched native D control/candidate pair, each 10 warmup plus
  10 measured complete optimizer updates, with the owner-requested saved D cache,
  actual configuration/activation, full series, TPS, memory and errors.
- [ ] One candidate diagnostic: 20 warmup updates, update 21, ranks 0/1,
  at most three complete microbatches while every rank completes accumulation 32;
  reuse the candidate timing cache.
- [ ] MCP Perfetto import reconciliation and scoped collective/order findings,
  original trace paths and hashes, compatible historical D mechanism comparison.
- [ ] Compact results JSON, reconciled status/planning, document checks and
  confirmation that task-owned GPU processes exited.

The control recipe is production-data D from
[the completed campaign](../production-data-abcd/campaign-plan.json), including
prefetch 32, BF16 expert-weight reuse, fused routing metadata, stable cap-zero CE,
FlashQLA `external_gate`, MTP and layer-0 FFN splits, and row-owner embeddings.
Keep real `data-002`, the production tokenizer, logical epoch 102400,
seed 2048, LR 65000/2000, B2/S4096/global batch512/accum32, EP8/FSDP8,
FP8/BF16/FP32 boundaries, model/loss compile, exact SAC, EMA, clipping and AdamW.
No MTP/FP8 A/A, embedding, checkpoint or whole-model numerical campaign is added.

## Execution route

The named SSH alias `sr008-gpu` failed DNS resolution (exit 255) on the first
read-only attempt. The shell is already inside the recorded owner allocation
`lm-mpi-job-23027817-4bfe-4730-926f-5c2de590b32b-mpimaster-0`; the owner explicitly
confirmed this direct route. Live checks found eight H100 80GB GPUs, zero compute
applications and zero used GPU memory, matching task NFS paths and the existing
`/opt/torchtitan-venv/bin/python3.12` runtime (PyTorch
`2.15.0.dev20260822+cu130`, CUDA 13.0). No new allocation or alternative host is used.

### Owner correction: reuse the saved compiler cache

During the first control's initial compiled update, the owner explicitly
requested using the already saved Triton cache. This supersedes the attachment's
fresh-cache policy. The first startup was stopped through its exact task-owned
process groups, with return code 130, one warmup update on all eight ranks and
no measured window. Its original runner receipts and logs remain unchanged.
The candidate source was restored exactly and all eight GPUs were released.

The saved completed D cache is
`runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache`.
It contains TorchInductor, Triton, DeepGEMM, HybridEP and TileLang caches;
all eight `triton/rank_0` through `rank_7` directories exist with hundreds of
entries each. Its producer completed successfully on the same pinned runtime.
Use the existing runner's `--compiler-cache-root` validation and this same saved
root for both timing runs. The diagnostic reuses the root after the completed
candidate has populated its changed kernels. All workload and measurement
settings remain as specified; the interruption authorizes no additional valid
pair, repeat or ablation.

The replacement control's actual metadata confirms `compiler_cache_policy:
reuse`, the saved root above and producer `proddata-d-timing-001`. Existing
producer/runtime/configuration validation passed. Its run ID is
`capacity-control-timing-002`; no interrupted-startup value is used for throughput.

## Results

The [focused correctness record](correctness.md) retains 13 group/Gloo checks,
9 dispatcher/gradient/Gloo checks, 12 CPU fullgraph cases, the real two-rank
Gloo compile/SAC check, and passing scoped Ruff/format checks. The payload
assertion was executed against the byte-exact original and failed on 128 versus
8 counters; the restored candidate passes. The [single independent review](review.md)
found no material source or test issue. The existing CUDA compile/checkpoint
selection also passes all 13 selected cases (82 unrelated cases deselected),
and three scoped Triton selector/fallback checks pass. No focused correctness
gate remains open.

The candidate communicates 8 int64 counters (64 bytes) instead of 128 (1024
bytes) at E128/EP8: 93.75% less tensor payload, with the same collective count.
Local expert counts and selection remain unchanged.

The [component measurement](component-results.md) completed one eight-rank
functional-NCCL process with 50 warmup and 200 alternating measured pairs per
scope. Every integer parity check passed, including counts above int32 and an
empty source with unequal peers.

| Event-inclusive latency, microseconds | Original A | Candidate B | Paired slowest-rank B−A |
| --- | ---: | ---: | ---: |
| Collective only | 78.960 | 78.976 | +0.128 |
| Complete aggregation fragment | 91.696 | 92.816 | +0.912 |

A/B columns are medians of per-pair rank maxima. The difference column is the
median of paired differences; it need not equal subtraction of the displayed
medians. These CUDA-event spans include functional-wrapper submission gaps,
communication and peer waits, and end at device completion. They do not isolate
NCCL kernel execution. No eager latency benefit is resolved; full distributions,
host scopes, tails, commands, hashes and the retained setup failure are in the
[component JSON](component-results.json).

The cached control completed all 20 optimizer updates on eight ranks. Updates
11–20 cover 20,971,520 physical input tokens and 20,966,400 supervised main
targets. The primary synchronized host envelope is 123.283711740 seconds,
yielding 170,107.79 physical and 170,066.26 supervised tokens/s globally.
Source/config guards pass, errors are empty and the child process group exited.
The candidate and trace remain pending; no A/B throughput conclusion is available.

After this valid control completed, the task wrapper incorrectly applied its
diagnostic receipt branch to orchestration role `control`, stopping before any
candidate launch. The native control itself passed. The failed wrapper state
and error are retained; correction must reuse this control and execute only the
remaining candidate and diagnostic. The valid null component result does not
cancel those requested measurements.

## Native configuration check

The coordinator compared the control's actual rank-0
`capacity-control-timing-001/native/config/resolved_config.yaml` with the
completed production-data D timing configuration using recursive comparison
after `yaml.safe_load`. The only changed field is `dump_folder`, pointing to
this experiment's unique output. All inherited workload, precision, compile,
SAC, input, optimizer, scheduler and feature configuration is therefore retained.
Candidate/control comparison and native activation/completion receipts remain
required before interpreting throughput.
