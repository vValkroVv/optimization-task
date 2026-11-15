# Native FFN component GPU results

**Status:** all eight component commands (two numeric checks, four unprofiled benchmarks, and two diagnostics) completed on eight H100 ranks. The component device ordering and overlap analysis is complete; see the [Perfetto evidence](ffn-component-perfetto.md). Combined native/head scheduling, full-training throughput, and production convergence remain open.

The raw JSON retains each per-rank sample series, slowest-rank sample, peak memory, buffer intervals, command, receipt path, and numerical comparison. The JSON is [ffn-component-results.json](ffn-component-results.json).

## Component summary

| Component | Numerical scope | Median rank latency, control → split (ms) | Allocated peak, control → split (GiB) | Reserved peak, control → split (GiB) | Unique data+grad storage/rank | Diagnostic |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `layer0` | Accepted scoped check; A/A exact, sparse split differences retained | 23.58189 → 21.56324 (8.56% lower split latency) | 3.026 → 2.731 | 3.938–3.996 → 2.902–2.902 | 1,214,547,072 → 1,214,547,072 (no growth) | [all 8 traces](artifacts/raw/ffn-components/layer0-diagnostic-001/rank0.trace.json) |
| `mtp` | User accepted observed MTP A/A variability; values retained | 31.41067 → 29.49934 (6.08% lower split latency) | 3.609 → 3.358 | 4.506–4.623 → 3.410–3.410 | 1,405,627,392 → 1,405,627,392 (no growth) | [all 8 traces](artifacts/raw/ffn-components/mtp-diagnostic-001/rank0.trace.json) |

Timing values are one sequential control/split pair per component. They are isolated block forward/backward latency measurements, not model tokens/s. No repeat or run-to-run noise estimate was collected.

## Numerical checks

Each check ran two 32-microbatch optimizer updates with native clipping and fused AdamW. Inputs/outputs, every logical parameter gradient, clipped/updated weights and optimizer state were compared. Initial logical FP32 parameters matched. Nonfinite counts were zero. No universal tolerance was imposed; the user directed that measured MTP/FP8 A/A variability be accepted as campaign variability without further investigation.

| Component | Comparison | Update | Phase | Unequal elements/rank | Changed tensors | Max abs | Max RMS | Max relative L2 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `layer0` | A/A | 1 | before_update | 0–0 | 0 | 0 | 0 | 0 |
| `layer0` | A/A | 1 | after_clip_adam | 0–0 | 0 | 0 | 0 | 0 |
| `layer0` | A/A | 2 | before_update | 0–0 | 0 | 0 | 0 | 0 |
| `layer0` | A/A | 2 | after_clip_adam | 0–0 | 0 | 0 | 0 | 0 |
| `layer0` | split_vs_A | 1 | before_update | 62–62 | 13 | 0.000488281 | 3.09026e-06 | 1.92687e-09 |
| `layer0` | split_vs_A | 1 | after_clip_adam | 119–119 | 31 | 9.31323e-10 | 3.63798e-12 | 2.80301e-09 |
| `layer0` | split_vs_A | 2 | before_update | 88–88 | 17 | 3.05176e-05 | 2.6974e-06 | 7.7182e-09 |
| `layer0` | split_vs_A | 2 | after_clip_adam | 208–208 | 38 | 3.72529e-09 | 1.49998e-11 | 1.30276e-08 |
| `mtp` | A/A | 1 | before_update | 3,997,766–4,007,413 | 13 | 0.125 | 0.0073764 | 0.000221195 |
| `mtp` | A/A | 1 | after_clip_adam | 8,589,305–8,589,305 | 33 | 0.00204 | 1.59375e-05 | 0.000722327 |
| `mtp` | A/A | 2 | before_update | 330,708,031–330,723,540 | 47 | 26.6542 | 8.31739 | 0.0808482 |
| `mtp` | A/A | 2 | after_clip_adam | 702,773,791–702,773,791 | 99 | 0.00208369 | 8.30782e-05 | 0.0604317 |
| `mtp` | split_vs_A | 1 | before_update | 234,247,372–288,544,085 | 35 | 7.25928 | 1.79758 | 0.0142371 |
| `mtp` | split_vs_A | 1 | after_clip_adam | 505,067,804–505,067,804 | 99 | 0.00204 | 0.0001275 | 0.125 |
| `mtp` | split_vs_A | 2 | before_update | 368,031,952–368,043,825 | 69 | 33.3005 | 10.4186 | 0.125 |
| `mtp` | split_vs_A | 2 | after_clip_adam | 702,795,446–702,795,446 | 99 | 0.00407956 | 0.000154472 | 0.0958386 |

For MTP, the first-update per-tensor A/A and split-vs-A metrics are in `numeric_checks.mtp.first_update_per_tensor_before_optimizer` in the JSON; this includes both input gradients, the MTP input projection/norms, and FFN w1/w2/w3. A/A differences were observed in those gradients and expanded after update 1. Per-user direction they are retained as accepted variability; no additional numerical run or tolerance policy was used.

## Buffer regions and actual component peaks

| Component | Control data / grad bytes | Split parent + FFN data bytes | Split parent + FFN grad bytes | Unique bytes/rank control → split | Allocated peak control → split (GiB) | Reserved peak control → split (GiB) |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `layer0` | 404,849,024 / 809,698,048 | 135,889,280@0 + 268,959,744@135,889,280 | 271,778,560@0 + 537,919,488@271,778,560 | 1,214,547,072 → 1,214,547,072 | 3.026 → 2.731 | 3.938–3.996 → 2.902–2.902 |
| `mtp` | 468,542,464 / 937,084,928 | 199,582,720@0 + 268,959,744@199,582,720 | 399,165,440@0 + 537,919,488@399,165,440 | 1,405,627,392 → 1,405,627,392 | 3.609 → 3.358 | 4.506–4.623 → 3.410–3.410 |

For both split components, rank inventories show the parent/remainder and `feed_forward` views share the original backing stores, partition them contiguously without overlap, and keep unique storage bytes equal to control. The child FQNs are `layers.0.feed_forward` and `mtp.layers.0.block.feed_forward`; each remains one GFSDP group containing native w1/w2/w3. The numeric memory peaks above come from the unprofiled benchmark sample windows. The profiler diagnostics do not supply peak-memory measurements.

The coordinator separately resolved the source-derived full-shape padded pool budget gate before these runs. Its reported ordinary FP32 key-None capacity is 2,101,346,304 B; MTP FFN capacity is 937,084,928 B. Both splits retain capacity, with residual 99,791,360 + FFN 134,479,872 elements at divider 64; the private head remains 2,101,346,304 B. The coordinator also reports the full-shape owned-gradient allocated/reserved peaks below baseline; those full-shape receipts are outside this component table.

## Activation and diagnostics

All receipts identify native `torchtitan.ops.float8.cuda_gemm.FP8Linear` for w1/w2/w3, BF16 compute with FP32 master/reduction, selective SAC, and Inductor model/loss compilation. `external_gate` activation was recorded on all ranks. Split INFO log markers were suppressed, so split activation is grounded in the resolved feature flag, the materialized separate FFN GFSDP group, disjoint region views, and diagnostic events.

### layer0

Each rank recorded 3 microbatch starts, 3 FFN input-gradient-ready markers, and 3 FFN post-backward reductions. All three FFN parameter gradients were ready before each reduction; the host ordering checks passed on all eight ranks. The separate [Perfetto analysis](ffn-component-perfetto.md#actual-device-overlap) confirms device ordering and compute overlap; host markers are supporting evidence only.

- [rank0 trace](artifacts/raw/ffn-components/layer0-diagnostic-001/rank0.trace.json) (1,914,510 bytes)
- [rank1 trace](artifacts/raw/ffn-components/layer0-diagnostic-001/rank1.trace.json) (1,914,776 bytes)
- [rank2 trace](artifacts/raw/ffn-components/layer0-diagnostic-001/rank2.trace.json) (1,912,499 bytes)
- [rank3 trace](artifacts/raw/ffn-components/layer0-diagnostic-001/rank3.trace.json) (1,890,731 bytes)
- [rank4 trace](artifacts/raw/ffn-components/layer0-diagnostic-001/rank4.trace.json) (1,914,694 bytes)
- [rank5 trace](artifacts/raw/ffn-components/layer0-diagnostic-001/rank5.trace.json) (1,915,487 bytes)
- [rank6 trace](artifacts/raw/ffn-components/layer0-diagnostic-001/rank6.trace.json) (1,914,481 bytes)
- [rank7 trace](artifacts/raw/ffn-components/layer0-diagnostic-001/rank7.trace.json) (1,914,657 bytes)

### mtp

Each rank recorded 3 microbatch starts, 3 FFN input-gradient-ready markers, and 3 FFN post-backward reductions. All three FFN parameter gradients were ready before each reduction; the host ordering checks passed on all eight ranks. The separate [Perfetto analysis](ffn-component-perfetto.md#actual-device-overlap) confirms device ordering and compute overlap; host markers are supporting evidence only.

- [rank0 trace](artifacts/raw/ffn-components/mtp-diagnostic-001/rank0.trace.json) (2,336,605 bytes)
- [rank1 trace](artifacts/raw/ffn-components/mtp-diagnostic-001/rank1.trace.json) (2,320,172 bytes)
- [rank2 trace](artifacts/raw/ffn-components/mtp-diagnostic-001/rank2.trace.json) (2,335,866 bytes)
- [rank3 trace](artifacts/raw/ffn-components/mtp-diagnostic-001/rank3.trace.json) (2,344,218 bytes)
- [rank4 trace](artifacts/raw/ffn-components/mtp-diagnostic-001/rank4.trace.json) (2,335,796 bytes)
- [rank5 trace](artifacts/raw/ffn-components/mtp-diagnostic-001/rank5.trace.json) (2,335,862 bytes)
- [rank6 trace](artifacts/raw/ffn-components/mtp-diagnostic-001/rank6.trace.json) (2,335,762 bytes)
- [rank7 trace](artifacts/raw/ffn-components/mtp-diagnostic-001/rank7.trace.json) (2,327,736 bytes)

Across all 48 rank/microbatch cases, the [Perfetto analysis](ffn-component-perfetto.md) confirms dInput and all three FFN weight gradients finish before packing, packing finishes before one FFN reduce-scatter, and that reduce-scatter precedes the parent reduce-scatter. The compute-only overlap interval spans 1.509–4.823 ms for layer0 and 1.560–2.939 ms for MTP. The normalized imports have exact device-event censuses and zero flow mismatches. The analysis does not retain event/stream handles, and each diagnostic synchronizes after its microbatch; it proves completed-reader reuse, not concurrent pending-reader reuse.

## Run identities, failures, and open gates

- Task HEAD: `2ada87125e433e1f76c4cb2bfad244b641a78a93`. Pre-cache-option harness SHA256: `fc57cff24713f285a675169a15956350e4f070981ff686812586fa1beb3aee1a`. Current harness SHA256: `878b73e5c4317515979112c6bcbcc663b86230b34b2b611bf72240a2db4a4654`.
- Native FLA/FlashQLA revisions: `1d2281ff1b7988d8070f00dba5ffa9747ae3a60e` / `a14dbf3474514e1674526431e960325f72081e9f`. The receipt pins exact task-local module paths and hashes.
- `layer0-check-001` failed before measurement because the first launch imported site-packages `fla`; the raw failure is [layer0-check-001.log](artifacts/raw/ffn-components/layer0-check-001.log). Corrected `layer0-check-002` passed on all ranks.
- The early layers.0 numeric check and later runs differ in one recorded source-file hash: `gift/torchtitan/distributed/gfsdp/row_owner_embedding.py`. Later MTP check and all control/split benchmark/diagnostic pairs share one identical 660-file map. The changed row-owner module is inactive in this component path because row-owner embedding is disabled and the component factory omits the embedding/table.
- The full per-rank sample series, slowest-rank series, peak values, exact launcher commands, receipt paths, and error summaries are in [ffn-component-results.json](ffn-component-results.json).
- Component buffer/peak measurements and device-overlap analysis are complete. The separate [Perfetto report](ffn-component-perfetto.md) retains the event-handle and per-microbatch synchronization limits. Combined native/head scheduling, throughput, full-training memory integration, convergence, EMA, and checkpoint continuation remain open; this table does not establish whole-model speedup or production readiness.
