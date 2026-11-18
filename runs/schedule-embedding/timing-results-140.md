## Seven-run native timing results

Primary time is the CUDA-drained single-host envelope across updates 41–140; mean update time is that envelope divided by 100. Physical and supervised TPS use the parser-validated global token totals.

| Variant | Receipt acceptance | Physical TPS | Supervised TPS | Mean update (s) | Max allocated (GiB) | Max reserved (GiB) | Retries / OOMs |
|---|---|---:|---:|---:|---:|---:|---:|
| A | ADJUDICATED_PASS (raw FAIL/1) | 149636.307 | 149599.774 | 14.014994 | 70.387 | 71.670 | 0 / 0 |
| B | ADJUDICATED_PASS (raw FAIL/1) | 156741.317 | 156703.051 | 13.379701 | 68.520 | 69.807 | 0 / 0 |
| C | ADJUDICATED_PASS (raw FAIL/1) | 160204.861 | 160165.748 | 13.090439 | 68.554 | 69.746 | 0 / 0 |
| D0 | ADJUDICATED_PASS (raw FAIL/1) | 162271.645 | 162232.028 | 12.923712 | 66.963 | 69.201 | 0 / 0 |
| S | ADJUDICATED_PASS (raw FAIL/1) | 163221.371 | 163181.522 | 12.848514 | 66.954 | 69.201 | 0 / 0 |
| SE | ADJUDICATED_PASS (raw PASS/0) | 163228.763 | 163188.912 | 12.847932 | 66.924 | 68.334 | 0 / 0 |
| SE-fused | ADJUDICATED_PASS (raw PASS/0) | 163154.589 | 163114.756 | 12.853773 | 66.962 | 68.352 | 0 / 0 |

ADJUDICATED_PASS rows retain the original raw wrapper FAIL/exit code and cite an external adjudication record, parser-summary hash, pinned repair-plan hash, source-freeze hash, method hash, and unchanged raw-receipt hashes in the JSON output.

### Incremental and cumulative comparisons

Each throughput and memory value is candidate/reference; update-time values are candidate/reference, so negative time change is lower elapsed time.

| Comparison | Physical TPS Δ | Supervised TPS Δ | Mean update time Δ | Max allocated Δ | Max reserved Δ |
|---|---:|---:|---:|---:|---:|
| B/A | +4.75% | +4.75% | -4.53% | -2.65% | -2.60% |
| C/B | +2.21% | +2.21% | -2.16% | +0.05% | -0.09% |
| D0/C | +1.29% | +1.29% | -1.27% | -2.32% | -0.78% |
| S/D0 | +0.59% | +0.59% | -0.58% | -0.01% | +0.00% |
| SE/S | +0.00% | +0.00% | -0.00% | -0.04% | -1.25% |
| SE-fused/SE | -0.05% | -0.05% | +0.05% | +0.06% | +0.03% |
| SE-fused/S | -0.04% | -0.04% | +0.04% | +0.01% | -1.23% |
| B/A | +4.75% | +4.75% | -4.53% | -2.65% | -2.60% |
| C/A | +7.06% | +7.06% | -6.60% | -2.60% | -2.68% |
| D0/A | +8.44% | +8.44% | -7.79% | -4.86% | -3.44% |
| S/A | +9.08% | +9.08% | -8.32% | -4.88% | -3.44% |
| SE/A | +9.08% | +9.08% | -8.33% | -4.92% | -4.65% |
| SE-fused/A | +9.03% | +9.03% | -8.29% | -4.87% | -4.63% |

### Cadence, cache and GC evidence

The JSON retains all 140 cadence records on every rank, full measured updates 41–140, per-rank first-20/last-20 statistics, measured ten-update bins, warmup bins 11–20/21–30/31–40, and per-update rank stragglers. The full-run plot includes startup intervals; update 1 is null because no preceding cadence exists. These host cadence intervals are not device-only times.

Cache boundary changes identify keys newly observed in (40,140]; they do not locate an exact compile update, distinguish compile from load, or establish execution for a pre-existing key. GC events are extracted from native logs; update attribution is left blank unless logged explicitly.

Plot generation: generated.
