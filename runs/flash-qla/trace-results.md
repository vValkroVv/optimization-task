# FlashQLA native trace comparison

## Result and accepted scope

Both new captures cover exactly three complete F/B microbatches on both ranks. Each rank has 18 original forwards, 18 SAC forwards, 18 VJP recompute forwards and 18 native backwards: 3F+B for six GDN layers × three microbatches. The GDN-specific outer coverage is 36 opaque forwards plus 18 recompute-VJPs. Actual Perfetto MCP import preserves all device-event counts/durations and unique host-flow links, with zero importer errors.

These are observed component work and overlap measurements. They do not establish full-update speedup. Each run trained afresh through update 21, so backend rounding and learned MoE routing can also change other work and contention.

## Native phase work

Kernel-duration sums in ms, accumulated across 18 calls per phase. Ranks are separate; streams are not summed into elapsed busy time. Outer scopes below account for gate work outside the external public Function.

| Rank | Variant | Original F | SAC F | VJP F | Native B | Native 3F+B |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | control | 8.965 | 12.902 | 8.900 | 28.858 | 59.625 |
| 0 | external | 3.748 | 6.549 | 3.775 | 16.524 | 30.596 |
| 0 | fused | 3.823 | 6.541 | 3.818 | 16.892 | 31.074 |
| 1 | control | 8.921 | 12.703 | 8.906 | 28.651 | 59.181 |
| 1 | external | 3.718 | 6.418 | 3.715 | 16.286 | 30.137 |
| 1 | fused | 3.783 | 6.537 | 3.815 | 16.854 | 30.989 |

## Whole GDN work budget

This table adds disjoint opaque original/SAC scopes and the GDN-specific VJP scope. It includes external gate forward/backward, casts and reductions owned by those wrappers. It excludes unproven compiled preparation outside them. Do not add native phase totals to this table.

| Rank | Variant | Control work ms | Capture work ms | Work removed ms | Control work/span | Removed work/span | Work ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | external | 59.651 | 31.422 | 28.229 | 4.75% | 2.25% | 1.898× |
| 0 | fused | 59.651 | 31.074 | 28.577 | 4.75% | 2.28% | 1.920× |
| 1 | external | 59.207 | 30.946 | 28.260 | 4.71% | 2.25% | 1.913× |
| 1 | fused | 59.207 | 30.989 | 28.217 | 4.71% | 2.25% | 1.911× |

The denominator is the same-rank control selected 3 MB device span. This small component budget explains why a roughly twofold component work reduction need not produce a comparable native training change. Work removed is not additive critical-path savings or an Amdahl prediction; SAC work overlaps communication and host/runtime behavior also matters.

## Integrated scope and idle

All values below are ms. Selected 3 F/B work is identified by host launch timestamps across all host threads. Full captured scope additionally includes setup/prefix/tail events; every excluded event and its proven shortest CPU owner is retained in each rank JSON.

| Rank | Variant | Selected events | Selected busy union | Selected device span | Selected idle | Full events | Full busy union | Full span |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | control | 9430 | 1220.910 | 1255.532 | 34.622 | 9445 | 1220.966 | 1263.573 |
| 0 | external | 9142 | 1194.498 | 1233.485 | 38.987 | 9157 | 1194.610 | 1241.717 |
| 0 | fused | 9088 | 1187.811 | 1225.778 | 37.966 | 9103 | 1188.178 | 1233.620 |
| 1 | control | 9430 | 1222.600 | 1255.716 | 33.116 | 9445 | 1223.377 | 1264.281 |
| 1 | external | 9142 | 1197.785 | 1232.961 | 35.176 | 9157 | 1197.844 | 1242.919 |
| 1 | fused | 9088 | 1186.883 | 1225.589 | 38.706 | 9103 | 1186.940 | 1234.043 |

Selected idle is selected span minus selected device union; the JSON also reports all captured-device busy/gap clipped to that span. Full captured scope is not a full optimizer update.

## Host work and overlap

Outer original/SAC+GDN-VJP scopes below are disjoint CPU call scopes. Host call and submission duration sums overlap device execution; they are not added to device time. All values are ms.

| Rank | Variant | Outer host calls | Outer launch APIs | Outer device union sum | Outer comm overlap union sum | Outer copy overlap union sum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | control | 52.250 | 3.065 | 59.669 | 12.787 | 0.000 |
| 0 | external | 35.824 | 1.926 | 31.439 | 6.689 | 0.000 |
| 0 | fused | 32.600 | 1.746 | 31.092 | 6.507 | 0.000 |
| 1 | control | 52.313 | 3.072 | 59.222 | 12.568 | 0.000 |
| 1 | external | 35.155 | 1.932 | 30.963 | 6.546 | 0.000 |
| 1 | fused | 34.352 | 1.813 | 31.006 | 6.534 | 0.000 |

Device union sum is a sum over disjoint outer scope groups, not the sum of kernel durations. Detailed per-call device unions/spans/gaps, host launch/call costs, named gate subsets and host wait API counts/durations appear in `trace-results.json` and rank analysis JSON. No event-specific wait dependency is proven. Timestamp overlap alone does not prove contention; collective kernels can spin while waiting.

Selected microbatch host waits follow. These are API wall durations, which overlap device execution, rather than extra latency to add to device spans. No synchronization API was enclosed by the narrower GDN outer calls.

| Rank | Variant | Selected host call sum ms | Stream synchronize calls / ms | Stream wait-event calls / ms | Event-query calls / ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | control | 1168.975 | 81 / 637.565 | 926 / 0.806 | 537 / 0.923 |
| 0 | external | 1150.834 | 81 / 625.397 | 926 / 0.817 | 558 / 0.944 |
| 0 | fused | 1143.078 | 81 / 618.568 | 926 / 0.809 | 561 / 0.980 |
| 1 | control | 1172.142 | 81 / 642.975 | 926 / 0.800 | 569 / 1.125 |
| 1 | external | 1148.117 | 81 / 633.201 | 926 / 0.788 | 576 / 0.981 |
| 1 | fused | 1145.693 | 81 / 602.680 | 926 / 0.831 | 504 / 0.905 |

## Mechanism and ownership

- The accepted FLA implementation already includes TileLang backward work; it is not a pure Triton baseline.
- External gate runs named `gdn_gate_fwd_kernel` outside the public QLA Function, followed inside by TileLang chunk-local cumsum, KKT solve and fused GDR forward. Actual MCP flow links demonstrate this nesting. Native backward owns prepare-H, fused GDR backward, GVA group reductions, fill and reverse cumsum; GDN-VJP additionally owns external gate backward and parameter-gradient reductions/casts.
- Fused gate moves gate+cumsum into `_FusedGate` before QLA solve/fused forward. Its actual backward regions include QLA backward and gate backward. Source pins identify the public QLA path and strict backend contract. Entry activation markers alone are not backward evidence. Actual MCP native-backward and GDN-specific VJP kernel counts/duration sums equal the raw ownership analysis on all four new traces.
- Original, SAC and VJP phases are derived from CPU ancestry plus correlation-linked launches, rather than kernel names alone. Generic `torchtitan::fla_vjp` includes conv/norm; the GDN-specific query requires enclosed native forward and backward.
- Anonymous TileLang kernels and compiled beta/sigmoid/contiguous preparation remain unassigned unless ownership or source symbols prove their role.

## Perfetto and reproducibility

Canonical raw files remain unchanged. The pinned normalizer only removes GPU user ranges and separates overlapping display lanes while preserving device timestamps, durations, names, arguments including actual stream, and host flow endpoints. Only normalized copies are transferred to the local MCP host. SHA256 receipts and event-by-event integrity checks are saved beside the copies.

| Variant | Rank | All raw/imported devices | Unique host-flow links | Missing/ambiguous | Import errors |
| --- | --- | ---: | ---: | ---: | ---: |
| control | 0 | 9445 | 9445 | 0/0 | 0 |
| control | 1 | 9445 | 9445 | 0/0 | 0 |
| external | 0 | 9157 | 9157 | 0/0 | 0 |
| external | 1 | 9157 | 9157 | 0/0 | 0 |
| fused | 0 | 9103 | 9103 | 0/0 | 0 |
| fused | 1 | 9103 | 9103 | 0/0 | 0 |

Actual load/SQL/results are under `artifacts/raw/trace-analysis/*-mcp.json`; accepted control rank0 evidence is reused, control rank1 and all four new traces were loaded through MCP. New normalized-file parity was checked against raw census and canonical-byte hashes. The raw analyzer SHA is `24b11673363ddedb2abe846b563f61962685b500fd9767c19e83a8c38696d662`; normalizer SHA is `4402b99ea328542009304f7d87723b5ff46ad3cbb9c22536fda3079fe5abe597`. `source-pins.json` records frozen backend/QLA/FLA identities. Capture controller and final `trace-capture-results.json` contain runtime/config/activation provenance.

## Limits

- Exactly one profiled capture per new backend and the accepted historical combined-four control; no repeated trace noise estimate or causal full-window backend comparison.
- Matched rank/microbatch/phase/call ordinal; native CPU frames do not record input dimensions. Frozen backend source guards the supported shape and precision. Tensor identity is not established.
- Each capture trains afresh through update 21. Backend rounding can change learned routing and MoE work/contention. Whole-window deltas are descriptive and are not isolated backend savings.
- Selected three F/B microbatches use launch ownership; full captured scope retains every raw device event. Neither covers the other 29 accumulation microbatches or optimizer.
- Kernel-duration sums may overlap across streams; device unions are elapsed busy time. Aggregate operation spans include intervening unrelated work. Per-call rows are primary.
- Outer opaque GDN and GDN-specific VJP scopes overlap inner native scopes. Never add their tables together. Generic fla_vjp also includes convolution and normalization and is not a GDN cost.
- Named gate/cumsum subsets do not assign anonymous TileLang or compiled beta/sigmoid/contiguous preparation. Source-backed named QLA cumsum/solve/fused kernels can be identified; unproven outside preparation remains unassigned.
- Temporal communication/copy overlap is not proof of contention, dependency or causal slowdown. Communication kernel duration may include spin/peer waiting.
- Host API durations and gaps are observable; no event-specific wait dependency is proven. No cross-rank clock alignment, full 32 MB critical path, optimizer gain, TPS gain, numerical parity or convergence follows from these traces.
