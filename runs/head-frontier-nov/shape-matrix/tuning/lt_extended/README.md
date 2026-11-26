# Extended 100B C8 cuBLASLt search

The bounded search found no reproducible improvement over native 5.5 for the isolated eight-chunk dW chain. The finalist led the first window by 0.315 ms on the graph timer and 0.559 ms on synchronized eager wall, using paired median differences. In confirmation it was 0.115 ms slower on the graph timer and 0.162 ms slower on eager wall against the same native control. The result retains the tested plan and samples without selecting a replay receipt. [Result](results/100b-c8-extended.json) · [Run log](run.log)

## Search and parity

The [earlier search](../lt/README.md) provided 16 heuristic seeds. The extension used the NVIDIA capability and configuration APIs to vary algorithm 66 tile, custom and cluster settings and applied `cublasLtMatmulAlgoCheck` to the live row-major and column-major descriptors. It stopped at 48 accepted additions per layout. Of the 698 new configurations checked, 96 passed and 602 returned `CUBLAS_STATUS_NOT_SUPPORTED`. Each passed plan used FP32 compute, one K partition, no reduction and at most four bytes of workspace. This was a bounded sample of the algorithm 66 neighborhood. [NVIDIA search example at a pinned revision](https://github.com/NVIDIA/CUDALibrarySamples/blob/a2eddd862e886c0ea5827013919801a1ddc25430/cuBLASLt/Common/LtMatmulCustomFind.h) · [cuBLASLt 13.1 API](https://docs.nvidia.com/cuda/archive/13.1.2/cublas/index.html#cublasltmatmulalgocheck)

The input is the existing production-tokenizer fixture with real `data-002` targets and packed positions, seed 2048 BF16 hidden states and weights, and BF16 dLogits from folded CUDA CE. Every write uses K = 2,048, V = 128,256 and H = 4,096 with one FP32 output slot. All 112 plans, including the 96 additions, matched native 5.5 bitwise over the full slot at both β = 0 and β = 1. The four shortlisted pairs also matched bitwise over the complete chain. Each replay activated one overwrite and seven accumulation writes. The finalist uses row-major tile 29 and custom option 1 for both writes, with cluster 4 for the overwrite and cluster 3 for accumulation.

## Chain timings

Both windows alternate arm order in separate graph and eager passes. Each graph sample measures one preallocated eight-write replay after capture and two warm replays. Each eager sample records submission, final wait and synchronized wall time. The first window has eight samples per arm. The confirmation window has twelve samples per arm and uses fresh graphs. The graph interval includes kernel execution and scheduling gaps. The eager interval includes Python submission and the final host wait. The measured region starts with operands ready and excludes projection, explicit copies and collectives. Individual GPU kernel and wait contributions were unavailable.

| Window | Arm | Graph median, ms | Eager wall median, ms | Submit median, ms | Final host wait median, ms |
|---|---|---:|---:|---:|---:|
| Discovery | Native 5.5 | 27.069 | 25.968 | 0.115 | 25.856 |
| Discovery | Native 5.5 repeat | 26.501 | 25.591 | 0.119 | 25.474 |
| Discovery | Finalist | 26.220 | 25.791 | 0.160 | 25.636 |
| Confirmation | Native 5.5 | 25.967 | 25.763 | 0.118 | 25.648 |
| Confirmation | Native 5.5 repeat | 25.904 | 25.867 | 0.117 | 25.742 |
| Confirmation | Finalist | 26.083 | 26.034 | 0.170 | 25.867 |

The paired repeat-minus-native differences in discovery were +0.065 ms graph and −0.221 ms wall. In confirmation they were +0.040 ms and +0.122 ms respectively. The paired statistics take the median of within-round differences, so they can differ from the difference of the table medians. The finalist and native control each used 6.806 GiB isolated GIFT total (0.749 GiB external, 6.057 GiB reserved, 5.996 GiB allocated peak), with zero allocator retries and 4 requested workspace bytes. This experiment measured a single dW chain on one H100. Integrated head and training timing require their own measurements.

## Reproduction

The recorded host used an H100 80GB HBM3, PyTorch `2.15.0.dev20260822+cu130`, CUDA 13.0, cuBLASLt 13.1.1 and NVIDIA driver `570.133.20`. The [launcher](run.sh) selects the supplied idle GPU by UUID and invokes the [search](search_c8.py). The receipt records the input and installed writer hashes, the serialized plans, timings, parity counts, activation and memory.

```bash
root=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125
bash "$root/tuning/lt_extended/run.sh" "$root" 1
```
