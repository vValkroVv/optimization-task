# Two chunk contributions per dW store

The paired writer retains the first chunk's BF16 logit gradient and hidden input. On the next chunk, one Triton TMA kernel computes two independent BF16-input, FP32-output dot products per output tile. It evaluates the same inter-chunk additions in order and stores the tile once. The existing FP32 GFSDP reduction slot, final reduce-scatter, dHidden operation and normal non-aliased weight gather are unchanged by this experiment. No concatenation, BF16 intermediate dW or split-K atomics are used.

| Fixed outer chunks | Current full head, ms / total GiB | Paired full head, ms / total GiB | Time change | Total change |
|---:|---:|---:|---:|---:|
| 2 | 76.214 / 11.638 | 82.006 / 13.591 | +5.792 ms | +1.953 GiB |
| 8 | 77.829 / 10.134 | 84.786 / 10.622 | +6.958 ms | +0.488 GiB |

Both C2 and C8 are dominated by the current writer on this isolated workload. Unlike the earlier concatenated-K test, all five full-output digests match, including every FP32 dW element. Holding the first dLogits of each pair across the next projection raises peak allocated memory by 1.957 GiB at C2 and 0.489 GiB at C8. Total includes allocator reservation and observed external device usage. Fewer global writes did not shorten the operation.

## Measurements and replay

- The [13 tile and pipeline configurations](pair.py) ran on real captured operands on H100 GPUs 7 (C2) and 6 (C8). Timing uses eight CUDA Graph replays after warmup and measures the complete dW chain with inputs ready. First-screen winning configuration 5 produced 29.107 ms at C2 and 31.443 ms at C8. The corresponding cuBLASLt controls bracketing this screen were 23.237/22.759 ms and 24.317/24.211 ms. The configurations with alternate register and pipeline settings were also slower. See [C2 screen](screen-c2.json), [C8 screen](screen-c8.json), [C2 additional configurations](tune-c2.json) and [C8 additional configurations](tune-c8.json).
- The sampled FP64 oracle covers 836 vocabulary rows across all 4,096 channels on actual BF16 inputs, including sampled target and vocabulary boundary rows. The best candidate and reference writer have equal relative L2, maximum error and BF16-rounding disagreement counts on that sample. The full native-head digest is an independent comparison.
- Full-head wall time includes target packing, folded main/MTP forward and backward, current in-place CE, BF16 dHidden, native singleton GFSDP, sharded gradient accumulation, hidden z-loss and host synchronization. Each fresh process ran 10 warmup, one priming and 10 measured calls; output hashes came from a separate reset call. Median of two process medians, maximum native GIFT total across both processes. GPU 6 ran control, pair, pair, control; GPU 7 ran pair, control, control, pair. [Verified summary](results.json) · [Native harness](../layout-order/replay_native.py)
- Standalone on the recorded host, run `bash run.sh 6 8` for the first kernel screen, `bash run.sh 6 8 tune` for the additional configurations, and `bash run.sh 6 8 control-a pair-a pair-b control-b` for the native replay. GPU 7 uses C2. The remote task folder is `/workspace-SR008.nfs2/vkropotin/head-frontier-nov/paired-dw`. Run [summarize.py](summarize.py) beside the downloaded JSON receipts to verify the measurements, modes, beta counts, full digests and actual pair-store counts.

The evaluation covers the isolated captured training head with one main and one MTP loss, one rank, and the two even chunk counts. It provides no measurement of full-training TPS.
