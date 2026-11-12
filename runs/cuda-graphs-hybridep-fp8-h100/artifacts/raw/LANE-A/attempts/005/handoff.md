# A005 native capture gate handoff

## Result

A005 failed with exit code 1 at 2026-11-12 00:22:43.514761 UTC (authoritative exit-status timestamp; launcher shell marker 00:22:43.475444642 UTC), before any optimizer update completed. Rank 4 had the earliest recorded step 1 forward/backward error at 2026-11-12T00:22:38.495332+00:00. All eight ranks recorded one forward/backward error and one step error; each also emitted a trailing forward/backward end and step end marker. There were zero successful optimizer step ends. The primary full traceback and all per-rank native error JSON files are retained under torchrun/ and summarized in failure-summary.json.

The separate read-only diagnosis in [LANE-A-PARTITION-POOL-DESIGN/001](../../../LANE-A-PARTITION-POOL-DESIGN/001/handoff.md) reports that partition_2 receives a nested buf47 HybridEP tuple; CUDA Graph Trees overlooks external input leaves and incorrectly treats eight aliases as pool-owned. It reports the dense GEMM safety tag is correctly placed. This attempt record preserves the failure evidence and does not retest the diagnosis.

## Run identity

- Source snapshot 004, base commit 446431dfa191ef0300f116257ae918bc85e38c06; manifest SHA-256 a17c9320dbf5862ff354992f4da104eaeb42011607feba8ffee58690226a8e30; archive SHA-256 9ebf6ca337bd6a80b9d691df654d60dc458af3514e574c444e87745cc2674f74.
- Preset gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml; SHA-256 b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b.
- Workload: 3B MoE, HybridEP EP8 and FSDP2 shard degree 8 on one eight-H100 server; global batch 256, local batch 8, sequence length 4096, accumulation 4, seed 42. The gate ran 25 steps against total schedule 30000; checkpointing disabled.
- Runtime: Python 3.12.3, PyTorch 2.15.0.dev20260822+cu130, CUDA build 13.0, NCCL 2.30.7, Triton 3.8.0+git3f6e4113, DeepEP 1.2.1+1b8f467, DeepGEMM 2.3.0+f093a4c.
- Graph environment: TORCHINDUCTOR_CUDAGRAPHS=1 and TORCH_LOGS=cudagraphs. Profiler requested skip 20, warmup 1, active 3, repeat 1, frequency 4. The launcher and observed torchrun command are recorded in manifest.json.

## Per-rank evidence

Every rank recorded one step start, four batch-fetch start/end pairs, one forward/backward start, one forward/backward error, one step error, and terminal end markers. No optimizer step start/end event appeared. All observer timing windows for steps 3–7 are incomplete, so no throughput value is valid.

Supervisor counts per rank: 8 CUDA Graph tree records, 7 warmups, 0 explicit replay lines, and 1 skip due to mutated inputs. Partition and unsafe custom-op diagnostics were also logged. These messages do not establish a successful capture or replay. No trace files were created.

## Cleanup and limits

By 00:23:19 UTC, runner, sampler, torchrun, all eight rank workers and observed Inductor workers were gone. All eight preflight H100 UUIDs showed 0% utilization and 0 MiB used; nvidia-smi listed no compute apps. The 500 ms memory sampler maxima are in failure-summary.json and are not exact peaks.

The one /proc/524786/maps read raced with unwind and returned ENOENT. CudaCherry GEMM shared-object path and SHA are unavailable and were not retried. Source, package and environment were not edited; there was no retry or relaunch. This failed gate cannot support a speed claim. Remediation and any future capture decision remain outside this execution record.
