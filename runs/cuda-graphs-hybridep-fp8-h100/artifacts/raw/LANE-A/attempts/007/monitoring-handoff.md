# A007 monitoring handoff

## Result

A007 completed successfully with exit code 0. The single launch started at 2026-11-12 01:32:38.989316 UTC and finished at 01:46:11.192707 UTC, 13m32.203391s later and about 88 seconds before its 15-minute deadline. It completed 25 optimizer updates on every rank. No retry or fallback launch occurred.

The launcher exit trap stopped and waited for the memory sampler. At 01:46:39 UTC, the exact launcher, torchrun, worker, and sampler PIDs were gone. The same eight NVIDIA H100 80GB UUIDs were each at 0 MiB used, and the compute-app query was empty. No manual process cleanup was needed.

## Run identity and configuration

- Launch invocation: [launch-invocation.json](launch-invocation.json); launcher: [launch.sh](launch.sh); lifecycle and exit status: [lifecycle.log](lifecycle.log), [exit-status.json](exit-status.json).
- Launch PID 547579; torchrun PID 547590; rank PIDs 547685–547692; memory sampler PID 547589. The launch metadata records a single launch and a 15-minute deadline.
- Source snapshot 006 has 92 frozen files. Its [manifest](../../../SOURCE-SNAPSHOT/006/manifest.json) SHA256 is 3a606c81d6b13692e422da04c333252620e504df9f1c4ba59c4bcf0c1a51f603; the source archive SHA256 recorded for this snapshot is 324cad36a75404f235f5dd7de10b571a75ad486276a44c09fe425f3187ebd51b.
- Preset identity is b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b. The launch metadata says the config overlay is identical to A006. Actual rank-zero resolved config: [resolved_config.yaml](native/config/resolved_config.yaml).
- The launch used the task-owned [observer.yaml](config/observer.yaml), with compile components model and loss, backend Inductor, and CUDA Graph exclusions for MoE layers.1 through layers.17. Selective activation checkpointing was enabled. Resolved values include local/global batch 8/256, sequence length 4096, accumulation 4, seed 42, FSDP2, EP8, and FP8.
- Runtime group membership from all-rank activation records: EP, FSDP, batch, loss, and world groups each contain global ranks 0–7 at degree 8; EFSdP has degree 1 at rank 0. CP, PP, and TP degrees are 1; DP shard degree is 8.
- Each rank’s activation record reports 18 activation-checkpoint wrappers, 18 compiled model-callable installations, 17 HybridEP dispatchers, and 17 sparse FP8 grouped-expert modules. The resolved config enables model and loss compilation. The callback has no separate compiled-loss execution counter.
- The 5-update timing-observer window covered steps 3–7 (start boundary 3, end boundary 8), reported complete/valid on all ranks, and recorded 1,048,576 global valid tokens per update on each rank. This is diagnostic-only evidence and is not used as a throughput claim. Each [rank observer record](native/timing_observer/rank-00000.jsonl) pairs an activation record with the timing window; corresponding files exist for ranks 0–7.

The exact torchrun arguments were:

    /opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 --log-dir=/workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/LANE-A/attempts/007/torchrun --redirects=3 --tee=3 /workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/LANE-A/attempts/007/compiler_entry.py /workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/LANE-A/attempts/007/config/observer.yaml --training.steps 25 --lr_scheduler.total_steps 30000 --debug.seed 42 --checkpoint.enable false --dump_folder /workspace-SR008.nfs2/vkropotin/optimization-task/runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/LANE-A/attempts/007/native --profiler.enable_profiling --profiler.profile_freq 4 --profiler.profiler_warmup 1 --profiler.profiler_active 3 --profiler.profiler_repeat 1 --profiler.profiler_skip_first 20

The launch set TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 before Torch import, TORCHINDUCTOR_CUDAGRAPHS=1, and TORCH_LOGS=cudagraphs. All eight [compiler-option markers](compiler-options/rank-00000.json) report Torch 2.15.0.dev20260822+cu130, mix_order_reduction=false, cudagraphs=true, and graph_partition=true. The marker outputs are available as rank-00000.json through rank-00007.json. Actual device identity was NVIDIA H100 80GB HBM3 on all eight devices; launch environment metadata contained an a100plus type string, which disagreed with the device and preflight identity.

## Completion and graph evidence

Each rank’s structured log reports 25 optimizer starts and ends, 25 step ends, 100 forward/backward starts and ends, one distributed teardown start and end, and zero ERROR/CRITICAL records. No structured event named finally was emitted. The successful optimizer-end count is therefore 25/25 for every rank; the run ended with teardown_end and exit code 0.

The CUDA Graph debug-log marker counts below are line counts from [supervisor.log](supervisor.log), not distinct graph counts. Warmup markers are Running warmup function; record markers are Recording function. Explicit Replaying function and Skipping markers were absent. The timing observer also says CUDA graph replay was “not observed by this callback” and marks runtime proof pending. These logs do not establish replay behavior.

| Rank | Warmup inference/forward/backward | Record inference/forward/backward | Explicit replay/skip lines |
| --- | --- | --- | --- |
| 0 | 1 / 5 / 9 | 1 / 5,623 / 1,529 | 0 / 0 |
| 1 | 1 / 5 / 9 | 1 / 5,418 / 1,505 | 0 / 0 |
| 2 | 1 / 5 / 9 | 1 / 5,712 / 1,536 | 0 / 0 |
| 3 | 1 / 5 / 9 | 1 / 5,712 / 1,536 | 0 / 0 |
| 4 | 1 / 5 / 9 | 1 / 5,041 / 1,489 | 0 / 0 |
| 5 | 1 / 5 / 9 | 1 / 5,414 / 1,516 | 0 / 0 |
| 6 | 1 / 5 / 9 | 1 / 5,371 / 1,509 | 0 / 0 |
| 7 | 1 / 5 / 9 | 1 / 5,562 / 1,522 | 0 / 0 |

The profiler created eight traces under [iteration_24](native/profiling/traces/iteration_24). Only paths and byte sizes were inventoried; trace contents were not opened.

| Trace | Bytes |
| --- | ---: |
| [rank0_trace.json](native/profiling/traces/iteration_24/rank0_trace.json) | 282,306,816 |
| [rank1_trace.json](native/profiling/traces/iteration_24/rank1_trace.json) | 281,796,106 |
| [rank2_trace.json](native/profiling/traces/iteration_24/rank2_trace.json) | 281,666,921 |
| [rank3_trace.json](native/profiling/traces/iteration_24/rank3_trace.json) | 282,484,694 |
| [rank4_trace.json](native/profiling/traces/iteration_24/rank4_trace.json) | 280,084,857 |
| [rank5_trace.json](native/profiling/traces/iteration_24/rank5_trace.json) | 282,260,623 |
| [rank6_trace.json](native/profiling/traces/iteration_24/rank6_trace.json) | 279,887,443 |
| [rank7_trace.json](native/profiling/traces/iteration_24/rank7_trace.json) | 280,313,456 |

Total trace inventory: 8 files, 2,250,800,916 bytes. The profiler used skip-first 20, warmup 1, active 3, repeat 1, and frequency 4. One profiled trace and the steps 3–7 diagnostic window do not establish a speedup.

## Memory, warnings, and remaining gates

[device-memory.csv](device-memory.csv) contains 12,896 samples: 1,612 per GPU from 01:32:39.273 through 01:46:10.633 UTC. The sampler queried nvidia-smi memory.used in MiB every 500 ms; this is sampled device-wide usage, not PyTorch allocated/reserved memory.

| GPU/rank | UUID | Peak sampled MiB |
| --- | --- | ---: |
| 0 | GPU-519db4ab-52c1-7730-5f03-bb8ecba718c3 | 36,733 |
| 1 | GPU-f0ff1e03-5cda-3f99-6b5b-22016cc776bf | 38,693 |
| 2 | GPU-35c5887c-b022-4a95-bdc4-f6c6e2175d52 | 35,411 |
| 3 | GPU-384c07d4-8c7d-6c6f-c4a8-66a7c18316eb | 34,869 |
| 4 | GPU-d52fef45-0944-a850-6d82-86acee633ce8 | 36,861 |
| 5 | GPU-9a1e0c80-7b57-12a4-6220-6a2b9a267466 | 35,959 |
| 6 | GPU-5f286803-df1a-a5db-5868-d31d1dde3201 | 35,523 |
| 7 | GPU-d03bb98d-079d-b834-e5a0-558188e55667 | 35,383 |

No fatal error occurred. Logs include Quack cache lock-timeout messages that fell back to in-process compilation without disk cache, and warnings that Triton key 1024 used fallback launch parameters. These are run conditions, not failure records. The supervisor log is 927,162,260 bytes; no full dump was copied into this handoff.

Open gates: CUDA Graph replay was not proven by the callback or explicit replay log markers; the FP8 custom-op execution marker explicitly says execution was not inferred from constructed module presence. No unprofiled A/A or A/B comparison or model-quality comparison was part of A007. Do not make a throughput or production-quality claim from this diagnostic run. Independent graph-mechanism analysis is owned under LANE-A/analysis/007.

## Evidence hashes

| Artifact | SHA256 |
| --- | --- |
| Snapshot 006 manifest | 3a606c81d6b13692e422da04c333252620e504df9f1c4ba59c4bcf0c1a51f603 |
| Snapshot 006 archive, as recorded for the launch | 324cad36a75404f235f5dd7de10b571a75ad486276a44c09fe425f3187ebd51b |
| compiler_entry.py | ffde3588d53012c7155fb5b6961a06be96095b44b8bc55c97a9808b2446ba065 |
| observer.yaml | 72b5c553c0ff51d25f685f469ce07723ba164ed6663128ad5632f103cd1791f8 |
| resolved_config.yaml | 30bbdcf5d915be0d53ef3a2ac7abfaad7e18d7ceda3dbd8265d50b3d31f34b6c |
| launch-invocation.json | 3a0edc3e5c91c090a82c4527b812afaa0f8f7f5baa173d1005f6ae83541468b9 |
| launch.sh | f7efdef0aee9341eaf1e7887d6da4499d59f39bb4e68b3492b95828e2ce51279 |
| device-memory.csv | c03ffd9304eca5cc298d9f2cc6c21663368a2da7331ca22fa10680b44de220df |

Compiler-option marker hashes:

| Rank | SHA256 |
| --- | --- |
| 0 | c2e99a3a4489f55dfb2eeb5ccbae5f743594a5ff2fb94b2c4633750087454548 |
| 1 | 68a8063f42a098c5e7b4bdb3aad1760fdeda45228d587d80bb285b7a3d389f86 |
| 2 | 5dd6e1968c791670a883d7e3adbe3d9bd3e5b55e3d5f9fcf95e42ff369bcfd3b |
| 3 | dcdab413ffba056e385ebfbd068806719eb6a0b0fa25983398a468a256e20040 |
| 4 | 58bde196acd7fb5708551d2d3be51a55f3b5a7150a9084a44f037bdaae2af37a |
| 5 | f13e155b388cbb4ebc6eb3a19e0c3f8aa1db32cc7bedf51db9f8b26b51636c67 |
| 6 | 5b2bce3444ae56835f7c9369f450057b7efc58433a2996589bc2b9a8f7e7a4dc |
| 7 | 03808eca71bbd3b4b78ff6770af17eab547f8aff586b0e8e077e8b9e48b97ada |
