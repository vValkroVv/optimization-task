# Physical-GPU thermal events in the C8 head replay

On SR008 job 60634e19, swapping physical GPUs 0 and 6 moved the late rank from logical 6 to logical 0 when the sustained 128k workload produced the slowdown. At the late trace, physical GPU 6 entered the FP32 reduce-scatter 160.67 ms after the earliest peer. The other seven took a median 1,242.87 ms from their first GPU work to reduce-scatter entry, versus 1,389.02 ms for physical GPU 6. The last arrival spent 6.34 ms executing its reduce-scatter. The prior all-gather finished within 0.076 ms across ranks. [Sustained replay phase receipts](../phase/phase-late.json)

The ordinary final-policy replay used the same process, physical device mapping, inputs, projections and collectives for all 32 synchronized calls, after ten warmups. Only sample 26 was profiled. We associated telemetry with the recorded wall-clock window for each rank, using the launch's explicit rank-to-UUID mapping. The next table uses unprofiled calls only. A thermal observation means the driver reported SW Thermal Slowdown or HW Thermal Slowdown on physical GPU 6. Counts are samples, not time integrals.

| Consecutive samples | Rank-max wall median | Thermal observations | Median of per-call median SM clocks on GPU 6 |
|---|---:|---:|---:|
| 1–5 | 1,255.92 ms | 6 / 42 | 1,372.5 MHz |
| 27–32 | 1,396.95 ms | 54 / 56 | 1,248.75 MHz |

Within profiled call 26, GPU 6 had 7 thermal observations out of 9. Its cumulative NVML thermal-policy time rose by at least 101.58 ms between readings strictly inside the call. Physical GPU 0, now logical rank 6, had zero SW/HW thermal observations in 287 readings over the 32 calls. Neither observed device reported HW Power Brake. Counter increases cover only intervals with both readings inside a call, so they are lower bounds. NVIDIA defines violationTime in nanoseconds. [Driver counter definition](https://docs.nvidia.com/deploy/nvml-api/api/structnvmlViolationTime__t.html) · [Matched driver observations](aligned.json) · [Step-level observations](steps.json)

With the same permuted mapping, the separate 10+10 candidate control had no sampled thermal events on GPU 6 and its unprofiled median was 1,256.01 ms, versus 1,360.43 ms for baseline in the same diagnostic pair. Separate arms may experience different thermal conditions. Aggregate throughput from the earlier acceptance grid remains a valid measurement of that run, but its timestamps lack per-GPU thermal records. The diagnostic identifies a reproducible mechanism without establishing an exact thermal correction for each historical row. Peak memory in the acceptance benchmark remains independently measured. [Swapped run receipts](../phase/phase-swap.json)

The all-device series began during the first 128k benchmark. The cause-specific flags and thermal counter started later and fully cover the swapped 128k experiments. An empty sampling window is missing evidence. The samplers used read-only NVML and nvidia-smi queries. All were stopped after the trials. [Broad sample](sample.sh) · [Thermal and power-brake reasons](sample-reasons.sh) · [NVML counter](thermal_counter.py) · [Rank alignment](align.py) · [Per-call alignment](step-reasons.py)
