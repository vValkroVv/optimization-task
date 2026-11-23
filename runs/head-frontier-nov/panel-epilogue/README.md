# Final dHidden store in the two-panel replay

The second vocabulary GEMM can add its FP32 result to the first FP32 dHidden contribution and round directly into BF16. This is a further test of the balanced, lower-memory panel schedule. The native panel control computes the same addition with FP32 `torch.addmm` and then casts the result.

Installed cuBLASLt 13.1.1 returned status 7 and zero algorithms for FP32 C with BF16 D on the BF16-input panel shapes. Both row and column descriptors were checked at C2/C8 for widths 32768 and 65536 and a 32 MiB workspace cap. The same probes returned algorithms with FP32 D. The [C2](probe-c2.json) and [C8](probe-c8.json) receipts preserve each query. The benchmark instead used the locally pinned Quack residual epilogue from `bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8`, with 128×128×64 tiles and a 1×2 cluster.

| Outer chunks | Native panel, ms / total GiB | Quack final dHidden, ms / total GiB | Difference |
|---:|---:|---:|---:|
| 2 | 88.665 / 10.739 | 90.594 / 10.817 | +1.928 ms, +0.078 GiB |
| 8 | 90.758 / 9.899 | 93.158 / 9.919 | +2.399 ms, +0.020 GiB |

All five full native-panel output digests, including dHidden and dW, remained identical. The Quack variant allocates a BF16 destination while the first FP32 contribution and panel logits workspace are still live. Removing a separate cast did not reduce peak total or improve wall time. These figures compare only two implementations of the same panel schedule. See the [standard-head comparison](../panels/report.md).

H100 7 ran C2 in candidate/control/control/candidate order; H100 6 ran C8 in control/candidate/candidate/control order. Each arm used the same Quack virtual environment and native panel caller, regular gather, real inputs, and ten warmup plus ten timed calls. Compilation, full-output digests and library-specific workspace probes were outside the timer. The first launch completed the timed operation but failed in the scalar-output digest helper; after correcting scalar byte views, fresh processes repeated every reported arm.

[Summary and hashes](results.json) · [Integrated wrapper](trial.py) · [cuBLASLt descriptor probe](mixed.py) · [Launch settings](run.sh)
