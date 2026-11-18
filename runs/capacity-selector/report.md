# GPU capacity selector on S

The owner accepted the GPU capacity selector for production on 18 November
2026. The selector-only result is the control for the later
[five-fusion comparison](../kernel-fusion/report.md#combined-five-at-40100).
This report compares S against S + GPU capacity selector. S includes
FlashQLA `external_gate`, the MTP and layer-0 FFN splits, row-owner embeddings,
and the GFSDP schedule fixes on top of the four fixes. Both runs used the
original 128-counter capacity all-reduce. The only training configuration
difference was `branchless_capacity_drop` in the two MoE block templates.

## Reference and expected cost

The SR008 timing and diagnostic receipts are `schedule-embedding-s-timing-140-004`
and `schedule-embedding-s-diagnostic-140-004`. They identify the reusable cache
as `runs/production-data-abcd/artifacts/raw/proddata-d-timing-001/cache`, with
rank-specific Triton directories. The original 128-counter `capacity.py` used
by the S launch adapter has SHA-256
`2f034b072fdaf02ac844936366134fafcd5e815588ce750b0dc7dfcc50467120`.

The existing selector carries the predicate as a CUDA tensor. In the common
no-drop branch, it writes a 65,536-byte output in place of the old `valid.clone()`
allocation. The affected path includes the same EP8 collective and the same
consumers of the mask. The component check must account for the selector's
launch and device time, possible copies from `contiguous()`, the compiled
65,536-route kernel's resource use, and any change in peak memory or SAC
behavior. The native comparison keeps the 50,000 fixed expert quota, SAC,
Inductor, FP8, and the S data and tokenizer settings.

The S rank-0 trace at update 21
([local shortcut](../../artifacts/4fix_qla_embed.json)) has 39 windows from
128-counter all-reduce completion to the start of FP8 input row quantization.
MCP Perfetto measured 14.302518 ms total, 0.550977 ms of recorded GPU activity,
and 13.751541 ms without recorded GPU activity across streams. At slice 24303,
the interval was 324.064 µs, including 310.144 µs without recorded GPU activity.
There are 39 one-byte predicate transfers inside the matched windows and 42
across the whole capture. These are control observations. The 13.586 ms estimate
from retaining GPU activity plus 30% overhead is a scenario to compare with the
candidate trace.

## Reproduction

From the task checkout on the authorized, idle eight-H100 master, run
`bash runs/capacity-selector/campaign.sh <new-run-id>` in a detached session.
The campaign records the real router inputs in a single S update, checks and
times the isolated section on EP8, runs the two native 10+10 timing windows,
and profiles the three-microbatch candidate window at update 21. All modes
retain the original LR schedule, seed, production data, and accumulation.
The S launcher validates the historical source, runtime, and warm cache.

Read `artifacts/raw/<new-run-id>-campaign/campaign.log` and `exit-code.txt`
under this run directory. Use the existing native timing summaries and
the component summary for measurements. Run [capacity window SQL](capacity_windows.sql)
and [predicate sync SQL](predicate_sync.sql) in MCP Perfetto on matched traces.
Use [trace_analysis.py](trace_analysis.py) to cross-check the original events
and compare collective and launch signatures.

## Execution status

On 2026-11-18, `sr008-gpu` reconnected after an initial gateway reset. The
owned master `lm-mpi-job-c316d31f-c997-4f8e-81ce-0d2c5fe4aea5-mpimaster-0`
had eight idle H100 80 GB GPUs on one node. The cache producer reported `PASS`
and its per-rank Triton directories were present. The original capacity overlay
matched its recorded hash.

Local checks: seven existing fixed-capacity group tests passed. Command
generation and override parsing verified the capture, timing pair, and trace
recipes against the historical S bridge on CPU.

## Isolated EP8 result

Run `capacity-selector-001-capture` completed one native S update and wrote
all eight real routing samples. `capacity-selector-001-component` passed on
all eight ranks at 65,536 routes per rank. The current conditional selector and
the GPU selector produced identical masks, masked IDs, masked scores, and score
gradients on the real sample and on the forced overload cases. The compiled
checks included selective checkpointing and both group gate outcomes, with
the same recorded collective order on each rank.

Each timing cell is the median of 50 observations. Each observation takes
the slowest rank in that repetition. The isolated calls start with empty GPU
queues. CUDA event elapsed time includes gaps between device submissions.

| Scope | Host elapsed S → candidate | CUDA elapsed S → candidate | Peak allocated S → candidate |
| --- | ---: | ---: | ---: |
| Selector | 80.286 → 62.171 µs (−22.56%) | 75.168 → 56.704 µs | 1,443,840 → 1,443,840 B |
| Preparation through routing metadata | 384.700 → 355.734 µs (−7.53%) | 378.912 → 349.824 µs | 7,474,176 → 7,474,176 B |

The peak reserved allocator reading was 69,206,016 B for both variants and
both scopes. Both selector calls added a 64 KiB allocation during execution,
with no allocation retained after output release. The original routing score
layout makes `weights.contiguous()` an alias. Separately, Inductor's mutation
probe conservatively treats the inputs as writable and generated a temporary
256 KiB FP32 score buffer with copy-in and copy-back in the compiled component.
This did not increase the measured peak. The direct selector reported 16 warps,
32 registers per thread, 1,024 B shared per CTA, and 658 spills in Triton's
kernel metadata.

The first isolated profiler initialized Kineto in CPU-only mode. Its eight
traces record one scalar read per control call and zero per candidate call.
The separate warm-cache `--profile-only` replay completed on all eight ranks
from `db6e8da9`. The preparation markers contain one scalar read, one one-byte
device-to-host transfer, and one `cudaStreamSynchronize` per control rank.
All three counts were zero for the candidate. The preparation kernel count
went from 10 to 11 on each rank. On rank 0 the new selector took 1.888 µs in
preparation and 1.856 µs as a standalone call. The candidate still ran the
existing all-reduce and routing metadata kernel. The two compiler-generated
score copies were fused into existing kernels. The profiler-only run and its
raw per-rank receipts are in `capacity-selector-001-component-profile`.

## Native 10+10 result

`capacity-selector-001-control` and `capacity-selector-001-candidate` both
passed. Each timed window covers 10 complete optimizer updates (11–20),
20,971,520 physical tokens, eight ranks, and 32 microbatches per update.
The global timer encloses all rank measurement boundaries.

The resolved configurations differ at `dump_folder` and the selector flag
under `gdn_moe` and `mla_moe`. The historical S diagnostic has the same
resolved training controls apart from run length, profiling, and output path.
The historical S and both timed runs share the same runtime and GIFT source
fingerprints. Startup and exit receipts for all 16 timed ranks attest the
original 128-counter overlay and each rank's reused Triton cache. The timing
pair ran from commit `c9a14a2a682370c7772a2f8d552f6d847f25f50f` with Torch
`2.15.0.dev20260822+cu130`, CUDA 13.0, and NCCL 2.30.7.

| Metric | S | S + GPU selector |
| --- | ---: | ---: |
| Total elapsed | 123.1297 s | 122.3006 s |
| Global physical tokens/s | 170,320.53 | 171,475.24 (+0.678%) |
| Mean optimizer update | 12.3130 s | 12.2301 s (−82.9 ms) |
| Median rank-0 host cadence | 12.3032 s | 12.2132 s |
| Maximum allocated across ranks | 65.8867 GiB | 65.8891 GiB (+2.48 MiB) |
| Maximum reserved across ranks | 67.5801 GiB | 67.5605 GiB (−20 MiB) |

Rank-0 host cadence in seconds, in update order:

```text
S          12.2558  12.2534  12.2719  12.2891  12.2777  12.3173  12.3332  12.3524  12.3700  12.4056
Candidate  12.1934  12.1715  12.2045  12.1978  12.2014  12.2372  12.2219  12.2688  12.2962  12.3058
```

All eight ranks reported zero allocator retries or OOMs. Cache observations
were stable in both measured windows. The host cadence includes the device
drain at the last boundary. Comparing median cadence in the last five updates
to the first five, S rose by 0.655–0.662% across ranks and the candidate rose by
0.564–0.582%. The reported gain comes from a single serial A/B and is preliminary.

The raw all-rank results and boundary receipts are under the two attempt
directories on SR008. SHA-256 of the respective `summary.json` files:

- S `329ae7826c5c11d7252fb21f799acbf1d12cd5536a477f56a74ac2717f32290a`.
- Candidate `d1557c9bc71b960ca9f6a0ca102a63d17b0cbf88f4f6a7997dfaff805baa2584`.

## Native trace mechanism

`capacity-selector-001-trace` passed and recorded three complete microbatches
on ranks 0 and 1 at update 21. The matched S diagnostics were previously
captured at the same update. MCP Perfetto and the original Chrome events give
the same capacity-window totals. The window begins at capacity all-reduce
completion and ends at FP8 input quantize start. The active and idle columns
use the union of recorded activity over all CUDA streams.

| Rank | Matched windows | Elapsed S → candidate | GPU active S → candidate | GPU idle S → candidate | Idle removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 39 | 14.3025 → 4.4608 ms | 0.5510 → 0.4917 ms | 13.7515 → 3.9690 ms | 9.7825 ms |
| 1 | 39 | 13.2285 → 1.7508 ms | 0.5452 → 0.4859 ms | 12.6833 → 1.2649 ms | 11.4184 ms |

The median elapsed window fell from 329.888 to 20.800 µs on rank 0 and
from 315.330 to 20.352 µs on rank 1. Across the full capture, each rank went
from 42 predicate transfers and paired host synchronizations to zero. Each
candidate rank executed 42 GPU selectors, with a median of 1.968 and 1.984 µs
respectively. The 39 capacity collectives kept their group membership, message
width, and relative sequence. The captures have matching quantize and dispatch
launch signatures, with quantize preceding dispatch at all 39 sites on each rank.

The original rank-0 example, S slice 24303 (collective sequence 35452), fell
from 324.064 to 21.216 µs elapsed and from 310.144 to 8.576 µs idle. Quantize
submission moved from 317.670 µs after all-reduce completion to 1.629 ms before
it. The GPU still executes the stages in dependency order. In the candidate,
the CPU began submitting the selector before all-reduce completion at 33/39
sites on rank 0 and 37/39 on rank 1. It submitted input quantize early at
33/39 and 36/39 sites respectively.

Six rank-0 and two rank-1 outliers explain 3.691 and 0.959 ms of the remaining
idle time. In those windows the short all-reduce finished 87–147 µs before the
CPU submitted the next stage. The trace shows `wait_tensor` queuing a CUDA
stream event wait. These gaps contain zero `cudaStreamSynchronize` calls.
Quantize starts within 2.141 µs of the corresponding host launch returning.
This points to late CPU submission after the GPU queue ran dry. The separate
8-byte and 576-byte transfers remain in both captures.

The candidate trace's total device span was about 19 ms longer on each rank.
Observed capacity all-reduce duration also varied between captures (rank 0:
11.221 → 16.134 ms, rank 1: 8.869 → 18.857 ms). The trace demonstrates removal
of the predicate wait and a substantial reduction in this local gap. Use the
unprofiled pair for the measured native throughput observation.

Candidate traces on SR008 are in
`capacity-selector-001-trace/native/profiling/traces/iteration_21/`. The
[original rank-0 candidate](../../artifacts/4fix_qla_embed_selector_gpu.json)
was copied byte-for-byte to the task's ignored artifacts directory. The
rank-1 trace and per-site original-event comparisons are in the local ignored
`artifacts/raw/trace-analysis/` directory beside this report.

| Candidate trace | Bytes | SHA-256 |
| --- | ---: | --- |
| rank 0 | 26,948,598 | `dbd9da0ddfb950c5c7f4e28e388bbac48879651596782d88945eab7fb687a5ad` |
| rank 1 | 26,953,540 | `9c20fbc58dce81c7deeaa2b279f3c35a5875e6334a04a05a363fc8342247fff5` |

## Production selection

Enable `branchless_capacity_drop: true` in both MoE block templates as an
explicit setting of the [production recipe](../../docs/baseline.md#current-final-task-baseline).
The exact drop-policy checks passed on all eight ranks, and Perfetto confirmed
elimination of the predicate host synchronization in the targeted intervals.
The native 10+10 pair measured +0.68% throughput. That figure is specific to
the measured pair. The shared library default is off.
