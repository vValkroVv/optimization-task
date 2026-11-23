# Isolated LM-head, MTP and loss frontier on H100

The exact fused CE was the only speed change to reproduce a lower whole-head latency with unchanged GIFT total memory at both fixed outer chunk counts. Vocabulary streaming provided lower-memory alternatives. C2 offered the larger tradeoff: eight vocabulary panels saved 1.641 GiB in the isolated head. The same change saved 0.410 GiB at C8.

## What was measured

Each comparison held the outer sequence chunk count fixed, with 8,192 folded token rows per outer chunk at C2 and 2,048 at C8. Each microbatch had 16,384 folded rows, local B2/S4096, D4096, V128256, one main and one MTP objective, and loss weights 1.0 and 0.3. Each fresh process ran one singleton native GFSDP head on an H100 and replayed captured production-tokenizer inputs, positions, targets and denominator. The native label packing, compiled loss boundary, head forward/backward, z-loss, BF16 dHidden assembly and FP32 sharded-gradient accumulation are included. Decoder layers, optimizer, and training throughput are outside the measurement.

The regular-gather control includes the previous fixes, early logits release, in-place BF16 logits-to-dLogits, direct FP32 cuBLASLt dW (C2 plans 9.14, C8 plans 5.5) and expandable allocator segments. The lower-memory historical control also aliases the BF16 gather send and receive storage; both sides of every alias comparison use that same policy. The campaign exercised all eight H100s on `sr008-gpu`. The captured-data replay worktree is `2c67c182`. The direct current-source checks used `lm10-c1`, HEAD `e9b7cdbb` plus the measured changes identified by per-file hashes. The regular controls produced the same five complete output digests on both checkouts at both shapes.

Tables report median synchronized host-wall milliseconds per complete isolated microbatch and maximum GIFT total GiB over measured calls. Each fresh process used ten warmup calls and ten timed calls after priming steady gradient accumulation. GIFT adds allocator peak reservation to sampled external device memory. Different same-shape lanes have slightly different control times; deltas below always use the control on that variant's own GPU and series. Methods and raw receipts are linked under each comparison.

## Speed at unchanged memory

The exact kernel stages the BF16 logits of a row across eight cooperating Hopper CTAs. It preserves the native 4,096-column reduction order, computes the FP32 loss, then writes the final scaled BF16 dLogits into the same allocation. Main and MTP scales are computed before rounding. The regular optimized control already reuses logits, which explains the unchanged peak. A simpler parallel sum changed BF16 gradient values and was rejected. Disassembly shows a single global logits-load traversal for the exact implementation. Physical HBM read bytes were unavailable from device counters.

| Fixed chunks | Gather | Exact implementation | Current head → candidate, ms | GIFT total both, GiB |
|---:|---|---|---:|---:|
| 2 | Regular | Ordered reduction | 76.153 → 75.569 | 11.638 |
| 8 | Regular | Ordered reduction | 78.506 → 77.709 | 10.134 |
| 2 | Regular | Ordered reduction + packed BF16 maximum | 75.900 → 75.314 | 11.638 |
| 8 | Regular | Ordered reduction + packed BF16 maximum | 78.487 → 77.160 | 10.134 |
| 2 | Aliased | Ordered reduction + packed BF16 maximum | 76.331 → 75.280 | 10.681 |
| 8 | Aliased | Ordered reduction + packed BF16 maximum | 78.767 → 77.454 | 9.177 |

The first two rows aggregate four C2 and five C8 matched pairs. The packed-max rows use two process pairs each per gather regime. All five complete output digests matched each control. Native FP32 per-row losses and BF16 dLogits matched in the direct chunk comparisons, including ignored labels and both vocabulary extremes. The route requires the tested immediate-backward loss scaling, two folded heads, current vocabulary width, and Hopper kernel support. It was implemented as a benchmark hook. [CE kernel comparisons, numerical checks, and individual timings](fused-ce/report.md) · [Independent replication on the current native source](native-ce-finalist/report.md)

### Confirmation on the current native source

A separate replay on H100 GPUs 0 and 1 installed only the CE hook into the current native consuming path. The five full outputs matched both the fresh native control and the historical control digests. The same native projection, weighted main/MTP loss, dW writer, GFSDP callbacks and z-loss were active.

| Fixed chunks | Native CE → exact ordered CE, ms | Change | Native CE → exact CE with packed maximum, ms | Change | GIFT total GiB |
|---:|---:|---:|---:|---:|---:|
| 2 | 75.803 → 74.787 | −1.34% | 75.678 → 74.738 | −1.24% | 11.638 |
| 8 | 79.532 → 78.100 | −1.80% | 78.823 → 78.482 | −0.43% | 10.134 |

Every adjacent native-versus-fused pair in this cross-check favored the fused arm. Each cell uses its own fresh control. A direct three-pair comparison of ordered versus packed-maximum CE measured 74.744 → 74.657 ms at C2 and 78.209 → 77.726 ms at C8. C2 differences were small (0.042–0.481 ms). C8 reversed direction in one pair, by 1.438 ms. The packed maximum has the faster CE-only kernel; the direct test did not establish a practical ranking for the whole head. [Complete independent series](native-ce-finalist/report.md)

## Memory at fixed outer chunk count

Panels divide vocabulary columns inside each outer chunk. The table lists the complete candidate time/total and the change from each same-series regular-gather control. Forward retains the last panel and the FP32 CE row statistics. Backward reprojects earlier panels, writes each panel's disjoint dW rows to the native slot, and combines dHidden contributions in FP32 before a single BF16 cast.

| Variant | C2 head ms / total GiB | C2 change, ms / GiB | C8 head ms / total GiB | C8 change, ms / GiB |
|---|---:|---:|---:|---:|
| Full vocab, two-panel-series control | 75.550 / 11.638 | — | 79.310 / 10.134 | — |
| 32,768 + 95,488 | 82.247 / 11.208 | +6.698 / −0.430 | 89.476 / 10.016 | +10.166 / −0.117 |
| 65,536 + 62,720 | 87.925 / 10.739 | +12.376 / −0.898 | 94.147 / 9.899 | +14.836 / −0.234 |
| Full vocab, four/eight-panel-series control | 75.798 / 11.638 | — | 77.917 / 10.134 | — |
| Four panels, width up to 32,768 | 96.120 / 10.251 | +20.322 / −1.387 | 101.942 / 9.782 | +24.024 / −0.352 |
| Eight panels, width up to 16,384 | 100.100 / 9.997 | +24.302 / −1.641 | 106.281 / 9.724 | +28.364 / −0.410 |
| Full vocab, paired-dW-series control | 76.214 / 11.638 | — | 77.829 / 10.134 | — |
| Full vocab, one dW store per pair of chunks | 82.006 / 13.591 | +5.792 / +1.953 | 84.786 / 10.622 | +6.958 / +0.488 |

A positive GiB change means higher peak total memory. Pairing dW reduces writes to the gradient slot while retaining the first chunk’s BF16 dLogits until the second arrives. The current writer was both faster and lower in peak memory at C2 and C8. [Paired writer receipts](paired-dw/results.json)

With the previous best aliased-gather stack, eight panels took C2 from 77.183 ms / 10.681 GiB to 102.479 ms / 9.040 GiB, and C8 from 80.097 ms / 9.177 GiB to 107.394 ms / 8.766 GiB. At fixed C8 the last move from four to eight panels freed only 60 MiB of total memory in either gather regime.

In the regular-gather numerical checks at all split sizes, the full captured BF16 projection, BF16 dLogits, integrated FP32 dW, weighted loss, z-loss and z-gradient matched their respective controls bitwise. Panel summation changes dHidden. On the same BF16 GEMM inputs, relative L2 against full FP64 was 0.00162925 for the native control, 0.00161692 for two balanced panels, 0.00161382 for four and 0.00161310 for eight. Accuracy against the FP64 oracle rounded once to BF16 also improved. These are local gradient checks on the captured batch. [Two-panel results](panels/report.md) · [Four/eight-panel results](multi-panel/README.md) · [Latency versus memory chart](memory-tradeoff.png)

## dW and dHidden experiments

| Change | C2 matched whole-head result | C8 matched whole-head result | Decision |
|---|---|---|---|
| Run dW before dHidden, with identical native GEMMs | 76.105 → 76.263 ms, same total | 79.208 → 79.602 ms, same total | Keep native order. The earlier 62–66 ms trace difference did not translate into a gain when only the order changed. |
| Physically pack hidden for dW, including copy in timer | 76.532 → 77.645 ms, +60 MiB total | Fresh reverse-order repeat 78.823 → 79.253 ms, +20 MiB | Keep original strides. The first C8 series showed a small apparent win that reversed. |
| Pack hidden and run dW first | 76.532 → 77.468 ms, same total | 80.057 → 79.973 ms, within spread, same total | No reproducible gain. |
| Preserve separate chunk contributions and store dW once per pair | 76.214 → 82.006 ms, +1.953 GiB | 77.829 → 84.786 ms, +0.488 GiB | Numerically exact on all complete outputs. Longer dLogits lifetime and slower GEMMs rule it out on these shapes. |
| Fuse final FP32 dHidden residual and BF16 store in the balanced panel route | Panel control 88.665 → 90.594 ms, +0.078 GiB | Panel control 90.758 → 93.158 ms, +0.020 GiB | The tested Quack epilogue reproduced panel results bitwise and was dominated by the native panel implementation. |

The layout trial evaluated 32 returned cuBLASLt configurations per shape, including both beta values. Nsight showed the dW + dHidden device sum rose for the order-only change on both shapes. The paired writer screened 13 Hopper TMA tile, pipeline and scheduling configurations on the captured inputs. Its full replay used the best screen result and retained exact per-chunk FP32 additions. The optional panel epilogue used Quack after the installed cuBLASLt rejected the mixed FP32 C and BF16 D descriptors. [Order and layout](layout-order/report.md) · [Paired dW](paired-dw/README.md) · [Panel residual epilogue](panel-epilogue/README.md)

## Decision

The exact fused CE is the speed candidate at the existing memory budget. C2 vocabulary panels are useful when a fixed-C2 run needs a specific additional memory margin. The asymmetric two-panel split is the least expensive of the tested memory reductions. At C8 the cost per GiB saved is considerably higher. Full-head latencies here describe one captured microbatch. They cannot establish a full-training TPS or memory change, or training convergence.
