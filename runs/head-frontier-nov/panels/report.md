# Two-panel vocabulary streaming

Both vocabulary splits lowered the isolated GIFT peak at fixed outer chunk
counts. The 65,536/62,720 split saved up to 0.898 GiB at C2 and 0.234 GiB
at C8, with 16–19% higher head latency. The 32,768/95,488 split gave up
roughly half the memory saving in exchange for less recompute.

## Regular all-gather

| Outer chunks | Vocabulary panels | Wall ms | Wall change | GIFT total GiB | Total change | Allocated GiB | Reserved GiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Full vocabulary | 75.550 | — | 11.638 | — | 10.348 | 10.354 |
| 2 | 65,536 / 62,720 | 87.925 | +16.38% | 10.739 | −7.72% | 9.454 | 9.455 |
| 2 | 32,768 / 95,488 | 82.247 | +8.86% | 11.208 | −3.69% | 9.911 | 9.924 |
| 8 | Full vocabulary | 79.310 | — | 10.134 | — | 8.834 | 8.850 |
| 8 | 65,536 / 62,720 | 94.147 | +18.71% | 9.899 | −2.31% | 8.610 | 8.615 |
| 8 | 32,768 / 95,488 | 89.476 | +12.82% | 10.016 | −1.16% | 8.724 | 8.732 |

[Per-process C2 receipts](regular-001/chunks-2/) · [Per-process C8 receipts](regular-001/chunks-8/)

## All-gather alias on both arms

| Outer chunks | Vocabulary panels | Wall ms | Wall change | GIFT total GiB | Total change | Allocated GiB | Reserved GiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Full vocabulary | 75.910 | — | 10.681 | — | 9.370 | 9.396 |
| 2 | 65,536 / 62,720 | 88.509 | +16.60% | 9.802 | −8.23% | 8.475 | 8.518 |
| 2 | 32,768 / 95,488 | 82.721 | +8.97% | 10.251 | −4.02% | 8.932 | 8.967 |
| 8 | Full vocabulary | 80.193 | — | 9.177 | — | 7.855 | 7.893 |
| 8 | 65,536 / 62,720 | 94.556 | +17.91% | 8.942 | −2.55% | 7.631 | 7.658 |
| 8 | 32,768 / 95,488 | 89.295 | +11.35% | 9.059 | −1.28% | 7.746 | 7.775 |

Descriptively, the conventional C8 control was faster and used less memory
than either streamed C2 point within each all-gather regime. The prior
aliased C16 replay reached 8.903 GiB at about 89.3 ms on another H100.
These comparisons do not establish a new overall frontier. The savings at
fixed C2 or C8 are paired on the same card. [Per-process C2 receipts](alias-001/chunks-2/) · [Per-process C8 receipts](alias-001/chunks-8/) · [Verified timing matrix](verified-summary.json)

## Numerical checks

Both splits matched the full-vocabulary BF16 logits and scale-weighted BF16
dLogits element for element in every outer chunk. All 525,336,576 entries
of the integrated FP32 dW matched the control digest in each case. The
weighted loss, z-loss, and z-loss gradient also matched bitwise. The
regular-gather control matched all five full-output digests of the
independent regular-gather replay of the current native source
`e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f` at both chunk counts.
Its matched wall medians were 76.105 ms at C2 and 79.208 ms at C8, with
identical GIFT totals to the regular controls above. [C2 native digests](native-control/c2.digest.json) · [C8 native digests](native-control/c8.digest.json)

| Relative L2 error | Full vocabulary | 65,536 / 62,720 | 32,768 / 95,488 |
| --- | ---: | ---: | ---: |
| BF16 dHidden vs FP64 | 1.62925e−3 | 1.61692e−3 | 1.61452e−3 |
| BF16 dHidden vs FP64 rounded to BF16 | 1.11233e−3 | 7.78816e−4 | 5.97694e−4 |
| C2 FP32 dW vs FP64 | 1.29492e−5 | 1.29492e−5 | 1.29492e−5 |
| C8 FP32 dW vs FP64 | 3.82294e−6 | 3.82294e−6 | 3.82294e−6 |

Both chunk counts produced the same per-variant dHidden errors. Compared
with the integrated control, the balanced and asymmetric splits changed
1,409,694 and 1,970,968 of 67,108,864 BF16 dHidden entries respectively,
with maximum absolute difference 5.821e−11 for both. The split changes
the FP32 GEMM and accumulation schedule before the one final BF16 cast.
Every output was finite. The FP64 gradient oracle used the full projection's
actual BF16 dLogits, hidden and gathered weight. The FP64 centered CE
oracle yielded 0.05988400133238 from the control BF16 logits. Integrated
loss was 0.05988400056958 in every arm. [C2 numerical checks](numerics-001/chunks-2/verify.json) · [C8 numerical checks](numerics-001/chunks-8/verify.json)

## Execution and scope

- The process-local experiment keeps the real main/MTP outer loop, original
  per-head scalar operations and metrics, native GFSDP gather and
  reduce-scatter, and independent hidden z-loss. It uses the preserved
  `lm-head-mtp-fix` replay at `2c67c182e8de6145f3796d62203714ab1b815446`
  and existing tuned cuBLASLt writers (C2 9.14 and C8 5.5). Both panels
  use the same respective algorithm serialization with their own output
  row ranges and dimensions. No source checkout files were edited.
- One BF16 workspace holds either panel. The FP32 centered maximum and
  denominator carry through the original 4,096-column reduction boundaries
  in order. Backward consumes the retained second panel first, then
  recomputes the first. Each panel runs dHidden before its dW. The same
  FP32 GFSDP slot receives the two disjoint vocabulary slices with the
  original per-outer-chunk beta. The original callback completes only
  after both writes. This adds 51.1% or 25.5% of one projection's work
  across a microbatch, with no additional collective in the replay. The
  schedule follows the pinned [vocabulary-panel Linear+CE reference](https://github.com/KohakuBlueleaf/KohakUwULLM/blob/d4a52f77c683e6a0622d76a4f42752d97ced97ba/src/kohakuwullm/kernels/loss/chunked_ce.py#L125-L237).
- The fixture came from the production-tokenizer training capture on SR008.
  It contains B2/S4096 with two heads, D4096, V128256, 16,384 folded
  physical rows, and 8,190 main and 8,188 MTP valid rows. The 2,233
  distinct valid main IDs range from 1 to 123,407. Outer chunks contain
  8,192 and 2,048 rows for C2 and C8 respectively. Both heads use the captured
  denominator 2,096,640, with weights 1.0 and 0.3 and observed FP32 backward
  scales 4.7695363e−7 and 1.4308610e−7. Capture SHA-256 is
  `15227882b1da44c2881b63622b6249bcddd307d90575216290f29ab9f720c727`.
- Each arm ran twice in fresh processes on the same physical GPU for that
  chunk count, GPU 0 for C2 and GPU 1 for C8. Each process used ten
  warmup and ten measured invocations. The process retained a primed FP32
  weight gradient in timed calls. Wall time is the median of the two
  synchronized host-wall medians. Within-arm spread was 0.14–1.29 ms,
  except the aliased C8 control at 2.33 ms. Timings include host submission,
  loss/gradient work, singleton NCCL work, and synchronization. Device
  compute, host work, waits, copies and communication have no separate time
  measurement in this experiment.
- Memory peaks are maxima over all twenty timed calls of an arm. GIFT total
  adds peak allocator reserved to device memory outside PyTorch sampled at
  the end of the call. The logged allocator and gather checks reported no
  timed retries or OOMs and 23 native gather API calls per process. CUDA
  Graphs were disabled. The control compiled the native CE forward with
  Inductor, and the candidate used isolated Triton kernels in the same
  eager outer loop. These are single-rank H100 replays with singleton SUM.
  Full-training runtime, convergence, and multirank candidate behavior
  remain outside the measurement.

The remote run root is
`/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/panels`.
It retains the full numerical tensor dumps. The local receipts preserve
timings, peak memory, activation, checksums, and every FP64 comparison.
The runtime was PyTorch `2.15.0.dev20260822+cu130`, CUDA 13.0 and
cuBLASLt 130101. [Replay](panel_replay.py) · [Timing launcher](lane.sh) · [Numerical checker](panel_numerics.py) · [Numerical launcher](numeric_lane.sh) · [Summary verifier](summarize.py)
