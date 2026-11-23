# Four and eight vocabulary panels

Outer loss chunking stays fixed at two or eight. Each outer chunk keeps all its token rows throughout. The vocabulary is split into widths of 32,768 or 16,384, with the final group covering the tail. Forward visits panels in vocabulary order and carries the native centered FP32 CE statistics and chosen logit across panels. Backward consumes the resident last panel first, then visits the earlier panels in reverse order and reprojects each once. Disjoint dW rows still receive each outer chunk's contribution via the direct FP32 writer. dHidden is accumulated in FP32 across panels and cast to BF16 once. Existing GFSDP buffers and the weight-gather policy stay the same within each comparison.

| Outer chunks | Gather | Vocabulary panels | Whole-head ms | GIFT total GiB | Total saving vs matched control |
|---:|---|---:|---:|---:|---:|
| 2 | Regular | 1 | 75.798 | 11.638 | — |
| 2 | Regular | 4 | 96.120 | 10.251 | 1.387 GiB |
| 2 | Regular | 8 | 100.100 | 9.997 | 1.641 GiB |
| 8 | Regular | 1 | 77.917 | 10.134 | — |
| 8 | Regular | 4 | 101.942 | 9.782 | 0.352 GiB |
| 8 | Regular | 8 | 106.281 | 9.724 | 0.410 GiB |
| 2 | Aliased | 1 | 77.183 | 10.681 | — |
| 2 | Aliased | 4 | 96.867 | 9.294 | 1.387 GiB |
| 2 | Aliased | 8 | 102.479 | 9.040 | 1.641 GiB |
| 8 | Aliased | 1 | 80.097 | 9.177 | — |
| 8 | Aliased | 4 | 103.774 | 8.825 | 0.352 GiB |
| 8 | Aliased | 8 | 107.394 | 8.766 | 0.410 GiB |

At both chunk counts all vocabulary panels reproduced the original full-vocabulary BF16 projection and dLogits elementwise on all captured rows. All five integrated outputs except dHidden matched the control bitwise. Relative L2 for dHidden against the full FP64 GEMM on the same BF16 inputs was 0.00162925 for the control, 0.00161382 with four panels, and 0.00161310 with eight. Against the oracle rounded to BF16, the corresponding errors were 0.00111233, 0.00053739 and 0.00037264. The independent panel-by-panel gradient exactly reproduced the integrated BF16 dHidden. dW retained the original FP32 writer and per-outer-chunk summation. [C2 full oracle](c2-oracle.json) · [C8 full oracle](c8-oracle.json)

Both arms in each gather regime were measured twice in fresh processes with ten warmup and ten measured invocations. Regular C2/C8 used H100 7/6, aliased C2/C8 used H100 2/3. The same real capture, loss, z-loss and native singleton GFSDP fixtures are used as in the [two-panel study](../panels/report.md). Numbers describe the isolated component. The cost of each additional panel includes an extra GEMM launch in each direction, more CE launches and a larger fraction of the vocabulary reprojected. Eight panels recompute 114,688 of 128,256 vocabulary columns. [Verified receipts](results.json)

The remote experiment folder is `/workspace-SR008.nfs2/vkropotin/head-frontier-nov/multi-panel`. [extend.py](extend.py) derives the replay and numerical checker from the two-panel experiment; [run.sh](run.sh) and [alias.sh](alias.sh) specify exact launch settings. [summarize.py](summarize.py) verifies the matched process groups, activation, dtype results, native memory readings, writer calls and numerical evidence.
