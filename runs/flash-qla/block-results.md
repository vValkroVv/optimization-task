# FlashQLA six-fixture block comparison

## Decision

Both authorized QLA native140-update performance comparisons may proceed against the common FLA control under the existing native plan: 40 warmup updates, 100 measured updates, common prefetch32/BF16 reuse/routing/CE and capture compiler cache. All 18 reports match the six fixture manifests and pinned backend/harness identities, all process receipts exit successfully with unchanged recorded source hashes, and the completed 15-run matrix preserves the complete source snapshot. No material numerical failure or new blocker appeared. This closes the bounded block-diagnostic gate; production acceptance remains open.

A = external_gate: existing FLA gate plus public pinned QLA (auto_cp=False). B = fused_gate: existing FLA gate/cumsum with natural-log convention plus thin pinned QLA orchestration. Accepted source and pilot reviews are reused; no new source audit or GPU experiment was performed.

## Evidence and contract

- task_revision: 0f32623d5e0e7544b20c1f047d94236cb6697758.
- fla_upstream_revision: 1d2281ff1b7988d8070f00dba5ffa9747ae3a60e.
- flash_qla_upstream_revision: a14dbf3474514e1674526431e960325f72081e9f.
- backend_sha256: 29780a731ef5fe70259511e38fc320f8d80227025eaa9eebb2878dbd17baa71e.
- harness_sha256: 2a143199c822e7b0a7054672b7617d5642f2f3f9fdf45a938c1765a8e7d9958b.
- [Completed matrix receipt](/workspace-SR008.nfs2/vkropotin/opt-fq/runs/flash-qla/artifacts/raw/controllers/block-matrix-001.json); [machine-readable synthesis](block-results.json) contains all report/receipt/manifest paths and SHA256, full metrics, samples and memory summaries.
- Six actual first-update rank0 VJP fixtures: B2/S4096/H16/HV32/KV128/GVA2, CP1, no initial/final state. Seven BF16 leaves include gathered A_log/dt_bias; FP32 gate arithmetic, captured BF16 sigmoid beta and upstream q/k normalization are preserved. Artifact shapes/strides are preserved; original storage offsets are explicitly rebased to zero.
- All 18 reports have finite output and seven gradients, exactly zero gradients for zero upstream, and exactly matching compiled/eager output and seven gradients. FLA replay output matches captured output exactly. These are observed fixture properties, not a universal allclose guarantee.

## Full sequence numerics

Relative RMS uses each quantity's reference RMS; absolute RMS and maximum absolute error use original tensor units. Cells give median [minimum, maximum] across six fixtures.

| Variant vs FLA | Quantity | Relative RMS (%) | Absolute RMS | Max absolute error |
|---|---|---:|---:|---:|
| A | output | 0.5095 [0.4408, 0.5184] | 5.459e-07 [4.357e-07, 5.746e-07] | 9.06e-06 [7.629e-06, 1.144e-05] |
| A | q | 0.5404 [0.4743, 0.5524] | 1.545e-10 [5.562e-11, 1.01e-09] | 5.588e-09 [1.863e-09, 5.96e-08] |
| A | k | 0.6183 [0.5868, 0.6467] | 1.962e-10 [8.524e-11, 1.295e-09] | 1.49e-08 [7.451e-09, 1.192e-07] |
| A | v | 0.3749 [0.3395, 0.4001] | 1.078e-09 [4.592e-10, 8.47e-09] | 1.49e-07 [2.98e-08, 1.907e-06] |
| A | raw_g | 0.5151 [0.4924, 0.5265] | 2.084e-11 [7.25e-12, 1.151e-10] | 1.164e-09 [4.657e-10, 3.725e-09] |
| A | beta | 0.493 [0.4505, 0.5087] | 1.827e-10 [7.968e-11, 1.093e-09] | 5.588e-09 [1.863e-09, 5.96e-08] |
| A | A_log | 0.363 [0.3187, 0.7569] | 5.317e-09 [2.508e-09, 2.311e-08] | 1.257e-08 [7.451e-09, 5.96e-08] |
| A | dt_bias | 0.3729 [0.3171, 0.7163] | 4.598e-09 [1.813e-09, 2.547e-08] | 1.304e-08 [5.588e-09, 5.96e-08] |
| B | output | 0.5095 [0.4408, 0.5184] | 5.459e-07 [4.357e-07, 5.746e-07] | 9.06e-06 [7.629e-06, 1.144e-05] |
| B | q | 0.5404 [0.4743, 0.5524] | 1.545e-10 [5.562e-11, 1.01e-09] | 5.588e-09 [1.863e-09, 5.96e-08] |
| B | k | 0.6182 [0.5869, 0.6467] | 1.962e-10 [8.523e-11, 1.295e-09] | 2.235e-08 [7.451e-09, 1.192e-07] |
| B | v | 0.3749 [0.3396, 0.4001] | 1.079e-09 [4.592e-10, 8.471e-09] | 1.49e-07 [2.98e-08, 1.907e-06] |
| B | raw_g | 0.5143 [0.492, 0.527] | 2.082e-11 [7.262e-12, 1.15e-10] | 1.164e-09 [4.657e-10, 3.725e-09] |
| B | beta | 0.493 [0.4499, 0.5091] | 1.825e-10 [7.97e-11, 1.093e-09] | 5.588e-09 [1.863e-09, 5.96e-08] |
| B | A_log | 0.3631 [0.268, 0.7569] | 5.319e-09 [2.508e-09, 2.311e-08] | 1.257e-08 [7.451e-09, 5.96e-08] |
| B | dt_bias | 0.3766 [0.3171, 0.7267] | 4.904e-09 [1.811e-09, 2.591e-08] | 1.304e-08 [5.588e-09, 5.96e-08] |

Fixture02 has the largest full-S parameter-gradient difference (A_log about 0.757%, dt_bias up to 0.727%). Its short-oracle parameter-gradient errors are comparable to FLA; this does not reveal a material numerical failure. All per-quantity A/B relative and absolute errors are retained in JSON. A/B differences remain small, without an established B benefit.

## Short FP64 recurrence comparison

Zero-state 64-token prefix only; relative RMS (%) median [minimum, maximum].

| Quantity | FLA | A | B |
|---|---:|---:|---:|
| output | 0.3395 [0.3272, 0.3428] | 0.3001 [0.2925, 0.3016] | 0.3001 [0.2925, 0.3016] |
| q | 0.376 [0.3672, 0.3835] | 0.3447 [0.3413, 0.3514] | 0.3447 [0.3413, 0.3514] |
| k | 0.4189 [0.4052, 0.4209] | 0.343 [0.3401, 0.3495] | 0.343 [0.3401, 0.3494] |
| v | 0.2915 [0.2825, 0.3064] | 0.2921 [0.2836, 0.3073] | 0.2921 [0.2836, 0.3073] |
| raw_g | 0.3429 [0.3256, 0.3462] | 0.2927 [0.2806, 0.3139] | 0.2926 [0.2806, 0.3139] |
| beta | 0.3011 [0.2833, 0.3349] | 0.3491 [0.3208, 0.3635] | 0.3491 [0.3208, 0.3635] |
| A_log | 0.3739 [0.2456, 0.3987] | 0.2885 [0.2237, 0.4093] | 0.2885 [0.2235, 0.4093] |
| dt_bias | 0.3578 [0.2312, 0.454] | 0.2859 [0.2225, 0.4566] | 0.2859 [0.2224, 0.4566] |

QLA output is closer to this short oracle across the six fixtures. Gradient quality is mixed: v/beta and some parameter-gradient fixtures can be slightly worse than FLA. Neither uniform numerical superiority nor full-S oracle accuracy follows.

## Device graph timing and inclusive memory

Single GPU0 H100 80GB; Torch 2.15.0.dev20260822+cu130/CUDA13.0. Same-input warm cache: 5 warmups, 10 samples, 10 graph replays per sample. Time is CUDA Graph device elapsed, excluding Python/autograd submission. Native B excludes preparatory F. Opaque VJP recomputes F (2F+B); synthetic SAC adds one F (3F+B). No full-step TPS is measured.

Cells give median [minimum, maximum] across fixtures. Spread is within-report maximum-minus-minimum sample time. Memory is MiB: inclusive process peaks within each phase, including live inputs/outputs, backward preparatory F and graph capture. Reserved values carry prior numerical/timing cache reservations; they are not incremental phase allocations or production savings.

| Phase | Variant | Time (ms) | Sample spread (ms) | Peak allocated (MiB) | Peak reserved (MiB) | Starting reserved (MiB) |
|---|---|---:|---:|---:|---:|---:|
| Native F | FLA | 0.4913 [0.4904, 0.4922] | 0.001459 [0.0007456, 0.002835] | 697.1 [697.1, 697.1] | 1908 [1908, 1908] | 1908 [1908, 1908] |
| Native F | A | 0.2127 [0.2117, 0.2141] | 0.000896 [0.0003904, 0.001722] | 386.1 [386.1, 386.1] | 1542 [1542, 1542] | 1542 [1542, 1542] |
| Native F | B | 0.2097 [0.2077, 0.2104] | 0.001122 [0.0006016, 0.003395] | 385.1 [385.1, 385.1] | 1540 [1540, 1540] | 1540 [1540, 1540] |
| Native B | FLA | 1.553 [1.546, 1.556] | 0.01217 [0.003018, 0.02067] | 1180 [1180, 1180] | 2056 [2056, 2056] | 2056 [2056, 2056] |
| Native B | A | 0.9218 [0.92, 0.9307] | 0.01107 [0.002518, 0.02108] | 730.6 [730.6, 730.6] | 964 [964, 964] | 964 [964, 964] |
| Native B | B | 0.9225 [0.9213, 0.9293] | 0.005672 [0.002042, 0.01386] | 730.6 [730.6, 730.6] | 962 [962, 962] | 962 [962, 962] |
| Opaque 2F+B | FLA | 2.522 [2.503, 2.548] | 0.04354 [0.02711, 0.0481] | 1341 [1341, 1341] | 2626 [2626, 2626] | 2626 [2626, 2626] |
| Opaque 2F+B | A | 1.351 [1.322, 1.362] | 0.03497 [0.02146, 0.04121] | 891.6 [891.6, 891.6] | 1380 [1380, 1380] | 1380 [1380, 1380] |
| Opaque 2F+B | B | 1.331 [1.317, 1.352] | 0.03227 [0.02126, 0.04016] | 891.6 [891.6, 891.6] | 1382 [1382, 1382] | 1382 [1382, 1382] |
| Synthetic 3F+B | FLA | 2.987 [2.986, 2.988] | 0.00289 [0.002086, 0.003693] | 1341 [1341, 1341] | 2786 [2786, 2786] | 2786 [2786, 2786] |
| Synthetic 3F+B | A | 1.569 [1.565, 1.572] | 0.005293 [0.002259, 0.006579] | 891.6 [891.6, 891.6] | 1540 [1540, 1540] | 1540 [1540, 1540] |
| Synthetic 3F+B | B | 1.559 [1.556, 1.562] | 0.004867 [0.001827, 0.008006] | 891.6 [891.6, 891.6] | 1544 [1544, 1544] | 1544 [1544, 1544] |
| Compiled opaque 2F+B | FLA | 2.503 [2.502, 2.504] | 0.002728 [0.001994, 0.003363] | 1341 [1341, 1341] | 2786 [2786, 2786] | 2786 [2786, 2786] |
| Compiled opaque 2F+B | A | 1.359 [1.355, 1.363] | 0.006166 [0.003178, 0.008157] | 891.6 [891.6, 891.6] | 1540 [1540, 1540] | 1540 [1540, 1540] |
| Compiled opaque 2F+B | B | 1.352 [1.348, 1.355] | 0.00709 [0.003142, 0.008614] | 891.6 [891.6, 891.6] | 1544 [1544, 1544] | 1544 [1544, 1544] |

Synthetic 3F+B FLA/QLA ratios of corresponding fixture medians: A 1.903 [1.901, 1.908]x; B 1.915 [1.914, 1.92]x. These support a useful component diagnostic signal without an established A/B benefit; they do not establish production speedup or memory savings.

## Limits and next measurement

- Six first-update rank0 boundaries, not post-warmup layer-labelled samples; strides preserved with storage offsets rebased to zero.
- 64-token zero-state FP64 recurrence is not a full-S oracle; no convergence, continuation, checkpoint or training-quality claim.
- Same-input warm-cache CUDA Graph device replay on one H100; excludes Python/autograd submission and backward preparatory forward.
- Host work, synchronization/idle/waits, copies and communication attribution unavailable; integrated operation/full-step latency unavailable.
- Synthetic 3F+B excludes model SAC schedule, sharding, communication, FP8 projections and optimizer.
- Memory is inclusive phase/process peak, including live input/output state, preparatory forward and graph capture; reserved memory includes carried allocator cache.
- Sequential fixtures/variants without repeated identical controls or alternating runs; ratios of medians do not establish general speedup or A/B superiority.

The authorized native140 runs are the next integrated magnitude experiment. Their resolved configurations, activation, steady-step series and allocated/reserved peaks must determine the full-step result. This matrix does not close distributed integration, continuation/convergence or production performance gates.
