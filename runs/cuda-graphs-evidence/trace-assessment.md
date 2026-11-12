# Preserved trace assessment

Extracted from the archived campaign report; historical 3B workload, not the current crop.

## CUDA Graph opportunity in the repaired baseline trace (12 November 2026)

**Assessment:** this captured update is predominantly limited by device execution and communication/dependency waits, with little exposed delay from CPU submission. The evidence lowers the priority of a substantial CUDA Graph implementation for this configuration. A short unprofiled comparison of an already working candidate could still establish a smaller benefit; this trace does not measure that benefit or authorize resuming experiments.

### Inputs and method

Analyzed the two update 111 Chrome/Kineto JSON files above through Perfetto MCP and PerfettoSQL. Rank 0 SHA256 is `5685a570d6c3959ecc4c55a4e739dbbc0ab809241e0d8bf1e31b24a8789f9391`; rank 1 is `fc206f05079df829e62cc0dab993b7ff67fca780e0aa41620915a02afdbb302a`. Server hashes were reconfirmed before recording this analysis. The preserved capture configuration and activation evidence remain applicable; no training/source change or new GPU run was made.

Device activity consists of `slice` rows with category `kernel`, `gpu_memcpy`, or `gpu_memset` and positive duration. The window runs from the first activity start to the last activity end. GPU annotations are excluded from occupancy. Busy time uses interval unions across streams; idle means no recorded activity on any stream in this window. Kernel names containing `hybrid_ep::` or `ncclDevKernel` identify communication families, including their device waits; other kernels are classified as compute. This classification does not separate communication transfer from spinning or peer waits.

Join `EXTRACT_ARG(arg_set_id, 'args.correlation')` between GPU events and both `cuda_runtime` and `cuda_driver` slices. All 15,781 rank 0 kernels have a matching API event, with unique API correlation IDs. For the main compute stream 7, compare API return against the preceding GPU event's end on that stream, including copies and memsets. Submission before that end establishes that the next kernel was already queued. A gap after early submission can still reflect dependencies, communication or device scheduling. Counts of compiled regions or short kernels alone do not establish CPU starvation.

### Observations

| Measurement | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| GPU activity span, including gaps | 3,106.797 ms | 3,106.810 ms |
| Main-stream kernels | 13,412 | 13,412 |
| API returned before preceding main-stream GPU event finished | 13,268 (98.9%) | 13,275 (99.0%) |
| Full-device idle within the recorded span | 135.094 ms | 135.204 ms |
| Largest full-device idle interval, at update start | 76.912 ms | 77.382 ms |
| Remaining idle after removing that interval | 58.182 ms | 57.822 ms |

Rank 0's median API-return-to-kernel-start delay is 126.492 ms (90th percentile 225.817 ms). This is queued/dependency latency, not CPU execution time. Neither `cudaGraphLaunch` nor `cuGraphLaunch` appears, and all recorded kernel graph IDs are zero: these are graph-off results.

The rank 0 elapsed-time partition is mutually exclusive: 2,238.815 ms with compute active (including 189.035 ms overlapping communication), 728.715 ms with communication active and no compute, 4.173 ms with only copies/memsets, and 135.094 ms idle. The 728.715 ms communication interval is 23.5% of the recorded span; its recoverable portion is unknown. These elapsed intervals must not be added to overlapping host or kernel-duration sums.

A concrete main-stream gap begins 462.564 ms after the first GPU event and lasts 8.016 ms. HybridEP `combine_kernel` occupies about 7.900 ms, followed by `device_sync_kernel` for 0.098 ms. The next compute kernel's launch API had returned 146.072 ms before the gap began. Faster submission of that already queued kernel cannot directly remove this gap; the trace associates it with communication/device waits rather than late CPU launch.

The 76.912 ms initial idle interval coincides with four DataLoader calls totaling 75.205 ms and ends with an HtoD copy. CUDA Graph replay does not perform this CPU data loading. Outside that initial interval, the 58.182 ms idle total is 1.87% of the captured span. Of this total, 39.118 ms consists of gaps at most 10 microseconds; for 38.545 ms of those gaps, the next activity's API had already returned before the gap began. The remaining full-device idle time before the next CUDA API call begins totals about 16.654 ms (0.54% of the span), including input preparation/copies. This is an observed timing category, not a claim that every such interval is capturable or removable.

Host API durations are distinct from exposed GPU idle. Rank 0 records 1,032.477 ms summed across 82 `cudaStreamSynchronize` calls, plus 186.656 ms in the final `cudaDeviceSynchronize`. These are host waits overlapping device progress, not an additional GPU-idle budget. The trace also records `Command Buffer Full` overhead events; launch-call duration therefore cannot be treated as pure CPU computation or directly converted into graph savings. Detailed Python self-time is unavailable because stacks were disabled.

### Decision and limits

The user's observation that the compiled workload normally keeps kernels queued is supported by the launch correlations on both ranks. Compilation and CUDA Graph replay remain different mechanisms. [PyTorch's CUDA Graph documentation](https://docs.pytorch.org/docs/2.15/notes/cuda.html#why-cuda-graphs) identifies bypassing Python/C++/driver dispatch as the main replay benefit and also notes possible device-side improvement. The present trace provides little evidence for a large gain from removing exposed CPU dispatch alone.

Prioritize understanding the exposed HybridEP/NCCL intervals and their transfer, dependency and peer-wait components before investing in broad graph capture. Keep the existing candidate available for a small authorized unprofiled A/B if desired. The 1.87% residual-idle fraction is neither a predicted speedup nor a strict end-to-end ceiling: graphs may also affect scheduling, device dispatch and distributed interactions.

This analysis covers one profiled update on two of eight ranks. Profiling changes host costs and rank timing; six peers are unobserved. It establishes neither steady-state TPS benefit nor a general negative CUDA Graph result. No throughput comparison, implementation change or further experiment was performed. CUDA Graph execution/performance research remains paused.
