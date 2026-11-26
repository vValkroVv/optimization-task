# Six-GPU head matrix thermal evidence

The sampler recorded 17,876 sweeps from 14:30:18 to 15:31:07 UTC on
26 November 2026. All 84 attempts across 21 cells ended during that capture.
All 76 successful attempts had covered measured windows and zero measured
thermal or power-brake flags. Eight OOM attempts left three cells without
complete timing comparisons. All eight cumulative thermal counters stayed
at their starting values. The maximum sampled active-GPU temperatures were
79 °C core and 78 °C HBM. GPUs 6 and 7 showed the idle reason on every sweep.
[Per-cell and per-attempt verdicts](verdicts.json) · [Sampler completion](samples.finished.json)

All 4,560 active-rank measured calls had counter brackets, with a maximum
counter-sample gap of 0.432 s. Of these, 3,702 had at least one direct clock-mask
sample. The software power-cap bit appeared in 16,368 of 16,415 active-GPU
measured samples. Fifteen attempts had one or more sampling gaps above the
one-second coverage threshold outside warmup and measured calls. The
longest was 1.868 s on unused GPU 6.

The sampler reads NVML every 0.2 seconds on all eight physical GPUs of
deployment `7c9acdc0-339d-480d-b7ea-9d3a43448e3a`. Ranks 0–5 map to physical
GPUs 0–5. GPUs 6–7 provide unused-device telemetry. The sampler writes epoch
nanoseconds on the GPU host. The full clock-event bitmask and separately
decoded event bits are sampled together. The reference timestamp for the
whole-thermal policy counter comes from NVML.

The driver exposed the whole-thermal policy and software power-cap counters.
The hardware thermal, software thermal and power-brake counters each
returned NVML error 2 on all eight cards. Their corresponding instant reason
bits were supported. Software power-cap activity and generic hardware
slowdown are reported independently of explicit thermal bits. The aggregate
thermal counter includes events from before this campaign. Use changes
between samples. See the [NVIDIA clock-event bit definitions](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html),
[thermal counter API](https://docs.nvidia.com/deploy/archive/R470/nvml-api/group__nvmlDeviceQueries.html)
and [counter timestamp structure](https://docs.nvidia.com/deploy/nvml-api/api/structnvmlViolationTime__t.html).
The [device temperature receipt](thermal-limits.txt) reports relative
T.Limit values, where 0 °C is the software operating threshold and −2 °C
is the hardware slowdown threshold. [NVIDIA SMI temperature fields](https://docs.nvidia.com/deploy/nvidia-smi/index.html#temperature)

The classifier uses reported process-level attempt bounds and each active
rank's exact host wall-clock call bounds from the rank receipts.
Warmup, measured calls, the rest of each arm and periods between attempts
are reported separately. GPU 6 and 7 use the union of rank call windows as
observation windows. Clock-event bits are sampled at a point in time.
Each cumulative counter increase is associated with the interval between
the previous and current counter timestamp. Intervals contained in a
measured call give direct evidence. Intervals crossing call boundaries
have ambiguous timing. Verdicts require valid counter brackets and gaps of
at most one second. A short call can have a valid counter bracket without
an observed in-call mask. The call medians are maxima across ranks for
each host elapsed span, then the median across measured calls.

The sampler, classifier and run results are stored beside the matrix
receipts on the verified SR008 shared filesystem. The raw `samples.csv`
is in the `thermal/` subdirectory. Its SHA-256 is recorded in the checksum
manifest. [Sampler metadata](samples.meta.json) · [Checksums](raw-provenance.sha256)

```text
/workspace-SR008.nfs2/vkropotin/optimization-task/runs/head-frontier-nov/shape-matrix-1125/length-grid/final-grid/gpu1-six-matrix/
```

Production environment, launch definitions, head activation and gradient
checks are recorded with the replay receipts. This telemetry assesses
thermal conditions in the isolated six-rank head workload.
