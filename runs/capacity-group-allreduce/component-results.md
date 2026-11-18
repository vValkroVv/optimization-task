# Capacity aggregation component measurement

One eight-rank H100/NCCL process completed successfully on the existing owner allocation. Reducing the tensor payload from 1024 to 64 bytes produced no resolved eager call-latency improvement in this bounded diagnostic. No training-speed or isolated NCCL-kernel claim follows.

## Timing result

All values are microseconds. Each variant has 200 samples per rank after 50 warmup pairs per scope. Pair order alternates AB/BA. Each sample starts after an excluded same-group barrier and CUDA synchronization and ends at actual CUDA event completion.

| Scope | A max-rank median | B max-rank median | Paired max-rank B−A median | Paired B−A p10 / p90 |
| --- | ---: | ---: | ---: | ---: |
| collective_only | 78.960 | 78.976 | +0.128 | -5.632 / +5.504 |
| fragment | 91.696 | 92.816 | +0.912 | -5.760 / +6.784 |

A fragment: functional SUM of 128 int64 counters, wait, then `view(8,16).sum(1)`. B fragment: `view(8,16).sum(1)`, then functional SUM of 8 int64 counters and wait. Collective-only uses precomputed inputs, excluding both local sums. Functional reduction allocates its output; those normal eager costs remain included.

CUDA event spans include host submission gaps, device communication and peer waits. They are inclusive call spans, not isolated communication kernel durations. The relocated local reduction may affect launches and compiler fusion; this eager diagnostic does not reproduce production model compile/SAC. Those gates belong to the native pair/trace.

The maximum is taken across ranks separately for each variant of each pair. Paired B−A is then formed from those maxima; it differs from subtracting displayed medians. There is no outlier rejection. Fragment maxima reached 2947.968 / 6311.808 us for A/B; all tails remain in raw samples and JSON.

## Per-rank event-inclusive distribution

Cells show median [p10, p90], in microseconds. Full min/p99/max and every host timing distribution are in [JSON](component-results.json).

| Rank | Collective A | Collective B | Fragment A | Fragment B |
| --- | ---: | ---: | ---: | ---: |
| 0 | 74.064 [71.424, 80.256] | 73.920 [71.008, 79.840] | 88.064 [85.088, 93.376] | 87.376 [85.184, 92.832] |
| 1 | 77.392 [74.880, 82.240] | 77.872 [75.008, 82.624] | 90.400 [87.904, 95.264] | 89.552 [87.552, 95.168] |
| 2 | 75.408 [72.544, 80.064] | 74.624 [72.160, 80.480] | 88.208 [85.984, 93.376] | 88.864 [86.560, 93.952] |
| 3 | 75.744 [72.704, 81.280] | 75.136 [72.416, 79.904] | 86.992 [84.928, 91.488] | 88.320 [85.824, 92.672] |
| 4 | 74.880 [72.512, 81.856] | 75.104 [72.224, 80.832] | 88.128 [85.952, 94.080] | 90.288 [86.112, 95.328] |
| 5 | 75.040 [72.320, 79.840] | 74.384 [71.360, 78.656] | 86.896 [84.576, 91.648] | 88.944 [86.048, 93.504] |
| 6 | 77.328 [74.784, 82.304] | 77.904 [74.784, 82.336] | 89.824 [87.936, 94.592] | 92.000 [88.320, 98.336] |
| 7 | 75.376 [72.000, 80.544] | 75.232 [72.256, 80.544] | 88.496 [86.080, 93.728] | 87.856 [85.888, 95.680] |

## Host submission and completion

Slowest-rank-per-pair medians in microseconds; host and event scopes overlap and must not be added.

| Scope | Submission A / B | Final event wait A / B | Total host A / B |
| --- | ---: | ---: | ---: |
| collective_only | 85.874 / 86.184 | 5.849 / 5.840 | 91.514 / 91.779 |
| fragment | 99.102 / 100.066 | 5.725 / 5.862 | 104.671 / 105.747 |

Host submission includes Python, dispatcher, eager allocation, wait-wrapper submission and event recording. Final host wait measures end-event synchronization. Separate device compute, communication-kernel duration and device idle attribution are unavailable here. There are no explicit input copies in the timed operation. Setup, inputs, parity CPU reads, warmup, barriers and pre-sample synchronization are excluded. No production memory, graph, convergence or throughput claim is made.

## Exactness and provenance

All ranks passed exact resulting group-load equality for bounded counters, counters above int32, and a zero-count source on rank 0 with unequal nonzero peers. int64 is preserved throughout, with results safely below int64 overflow. Raw parity loads are retained. These synthetic counter cases are independent of tokenizer/data values and do not substitute for the focused route-mask tests or real-data native comparison.

Runtime: `2.15.0.dev20260822+cu130`, CUDA `13.0`, NCCL `[2, 30, 7]`, `NVIDIA H100 80GB HBM3`. Process group ranks 0–7 match the native EP8 rank membership. Host: `lm-mpi-job-23027817-4bfe-4730-926f-5c2de590b32b-mpimaster-0`. Source revision `e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f`; exact script, installed functional-collective source and capacity hashes are in JSON. Installed `all_reduce` calls `_c10d_functional.all_reduce`; `AsyncCollectiveTensor.wait` invokes functional `wait_tensor`, which supplies CUDA stream dependency semantics.

Command:

```bash
source /etc/profile.d/wm.sh
/opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 runs/capacity-group-allreduce/component_benchmark.py --output /workspace-SR008.nfs2/vkropotin/optimization-task/runs/capacity-group-allreduce/artifacts/component-001 --warmup 50 --pairs 200
```

Exit code 0. Post-exit live `nvidia-smi --query-compute-apps` was empty. Raw per-rank sample JSON and combined log are outside tracked evidence under `artifacts/component-001/` and `artifacts/component-001.log`; exact paths/hashes are in JSON. `py_compile` and scoped Ruff F821/F811/F841 passed. Root independently reviewed timing/order before execution.

Retained setup failure: the first shell log redirection failed because the task artifacts directory was absent; no torchrun process started. Creating that directory fixed the defect. The sole actual measurement process completed. Nonfatal retained warnings: pynvml deprecation and IPv6 socket errno97 at rendezvous. No package, shared environment or training-source change was made by this benchmark.
