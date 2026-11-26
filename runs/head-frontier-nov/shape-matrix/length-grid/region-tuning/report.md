# cuBLASLt and CUDA CE on long-context LM-head chunks

These component replays target the existing GIFT LM head at local batch one, main plus MTP1, eight loss chunks and vocabulary 128,256. The 100B and 700B GIFT 3.5 linear heads have widths 4,096 and 7,168. Each measured chunk has one quarter as many folded rows as positions in the local sequence. Contexts of 32k, 64k, 128k and 256k therefore exercise 8,192, 16,384, 32,768 and 65,536 GEMM rows, respectively.

The control is the path actually used at that length on the replacement eight-H100 node. At 100B/32k the control already dispatched calibrated projection and weight-gradient plans. At the other seven shapes it used the ordinary PyTorch projection and the generic direct FP32 cuBLASLt writer. All eight used ordinary PyTorch dHidden and the tuned consuming CUDA CE. These controls follow the saved rank receipts. [Full-head length sweep](../report.md) · [Operating points and actual routes](../server-60634e19-summary.json)

Projection and dHidden are different GEMMs. Projection takes the folded BF16 hidden chunk and produces BF16 logits; dHidden takes the weighted BF16 dLogits and produces a BF16 hidden gradient. The weight gradient takes BF16 operands and accumulates into an FP32 slot. Its first chunk overwrites the slot and the remaining seven update it. CE receives the already-materialized logits and replaces them with weighted dLogits while producing separate loss sums.

Each group uses dedicated H100s on SR008 job `60634e19-cb86-4e8b-a27c-60078635cd4b`. The input data come from the B1 length replay, which repeats the original 4,096-position tile and preserves the packed boundaries. Each timing is scoped to its component; the complete distributed head, collectives and training step are outside the measurements. A regional peak includes the buffers needed to isolate that operation. Use the paired GIFT total within a region to compare memory.

## Results

The table uses synchronized wall milliseconds and region peak GIFT total in GiB. An arrow shows the control followed by the recommended plan. A single time keeps the existing route. Projection, dHidden and CE cover one representative chunk. The dW timer covers all eight writes. The memory in a cell applies to both plans. CE retains a pristine logits copy for its restore operation, while dW reuses one pair of operands at the same addresses over the eight writes. These different live sets make the peaks specific to their respective harnesses.

The linear controls combine the two identical control arms. Three borderline linear cases use the fresh 24-round head-to-head measurements. The dW controls use the primary control arm of the 12-round confirmation. The summaries show marginal medians. The selection tests use within-round differences, so their reported savings can differ from the subtraction of the displayed medians.

| Model | Sequence | Projection · ms / GiB | dHidden · ms / GiB | dW, eight writes · ms / GiB | Current CE · ms / GiB |
|---|---:|---:|---:|---:|---:|
| 100B | 32k | 11.63 / 3.720 | 11.39 / 3.778 | 94.41 / 4.749 | 1.73 / 4.657 |
| 100B | 64k | 23.94 → 23.43 / 5.731 | 23.09 / 5.849 | 196.54 / 6.761 | 3.27 / 8.544 |
| 100B | 128k | 48.02 → 46.66 / 9.774 | 49.40 / 10.028 | 387.80 / 10.804 | 6.34 / 16.376 |
| 100B | 256k | 96.99 / 17.841 | 99.16 / 18.349 | 775.80 → 764.98 / 18.909 | 12.47 / 32.040 |
| 700B | 32k | 20.91 → 20.37 / 4.499 | 20.76 / 4.599 | 179.31 → 167.96 / 6.272 | 1.74 / 4.638 |
| 700B | 64k | 42.73 / 6.550 | 43.07 / 6.786 | 352.89 → 333.78 / 8.362 | 3.27 / 8.583 |
| 700B | 128k | 85.24 / 10.692 | 87.04 → 83.35 / 11.122 | 704.06 → 667.73 / 12.464 | 6.35 / 16.415 |
| 700B | 256k | 172.13 / 18.954 | 175.14 → 165.95 / 19.833 | 1477.83 → 1360.96 / 20.765 | 12.48 / 32.059 |

At 700B, keep the new dW plan pairs for all four lengths, the projection plan for 32k and dHidden plans for 128k and 256k. At 256k the dW chain's median fell by 116.88 ms (7.9%) and the dHidden median by 9.19 ms per chunk (5.2%). For 100B, keep projection at 64k and 128k and the dW pair at 256k. The 100B/32k control already selected calibrated projection and dW plans; this search found no additional win there.

The five winning dW pairs all use algorithm 66 with four bytes of workspace, but their tile, cluster or matrix description can differ by shape and by overwrite versus accumulation. Each pair produced the same FP32 output bits after each of eight writes as the current path. All retained projection and dHidden plans produced the same BF16 output bits as their exact-shape control on the replay inputs. Two alternatives remain provisional after a fresh comparison: 100B/64k dHidden won 19 of 24 paired rounds, and 700B/64k projection gained 1.40 ms in paired wall time with 0.93 ms of A/A variation. Both stay out of the recommended manifest. The current tuned CUDA CE remains the fastest reproducible choice across this grid.

The operating point with the clearest combination for the next full-head test is 700B/256k. Apply only the shape-matched recommendations in the [linear manifest](linear/linear-plans.recommended.json) and add the [dW entries](dw/selected-catalog.json) to the prior catalog. Whole-head latency, memory and throughput with these new choices are still unmeasured.

[Projection and dHidden trials](linear/report.md) · [dW trials](dw/README.md) · [CE trials](ce/README.md)
