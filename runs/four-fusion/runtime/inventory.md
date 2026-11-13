# Four-fusion runtime route inventory

Captured 2026-11-13T17:47:59.957625+00:00 UTC on lm-mpi-job-8dff1e19-674b-42ed-96a9-bcd042dfefe5-mpimaster-0. This is read-only evidence; no training run, source edit, installation, or job change was performed.

## Allocation and paths

The active tool session is already on authorized master lm-mpi-job-8dff1e19-674b-42ed-96a9-bcd042dfefe5-mpimaster-0 for allocation lm-mpi-job-8dff1e19-674b-42ed-96a9-bcd042dfefe5. The master host file lists one node; the worker host file is absent. The task checkout is /workspace-SR008.nfs2/vkropotin/optimization-task on NFS4 (pd62-nfs-002.sr008.aicloud.sbercloud.tech:/vol_fg_002/namespace_ai0001071-01058). Eight H100 80GB devices were visible, each at 0% utilization and 0 MiB used; the compute-app query was empty. No task training/runner processes were observed. PID 1 is the allocation's mpirun wrapper and remains untouched. This is a point-in-time check; platform API status was not queried.

## Source, runtime, and data pins

At inventory capture, task HEAD was da6774a1c2191bfec0e1cb4aa1dc34019b3b7df8. Root later fast-forwarded origin/main to 33b336a592746340d327d86e34149a23dc8688ee; root reports only docs/profile-my-ideas.md and .gitignore were added and GIFT bytes are unchanged. The worktree remains dirty from concurrent four-fusion changes. Use each run's source_snapshot_before and source-final.json to pin the exact candidate tree. The accepted V2 provenance is recorded in pins.json.

Interpreter: /opt/torchtitan-venv/bin/python3.12, Python 3.12.3. PyTorch 2.15.0.dev20260822+cu130, torch.version.git_version 8b353065ce260336c5fdfa74d754e8bd9b91b062, CUDA build 13.0; Triton 3.8.0+git3f6e4113; DeepGEMM 2.3.0+f093a4c; DeepEP 1.2.1+1b8f467; CUTLASS DSL 4.4.1; Mosaic Streaming 0.13.0; Transformers 4.57.1; Tokenizers 0.22.2. Existing V2 native receipts report NCCL 2.30.7; it was not re-queried in this non-CUDA-initializing probe.

The live runtime exposes torch.nn.functional.linear_cross_entropy with the signature and source hashes in pins.json. The actual 29 MDS shard hashes and index hash all match runs/overnight-streaming-perf/mds-data.json. The fixture contains 896 rows and 3,670,016 encoded tokens. Each arm must use epoch_size=51200 and prefetch depth 32.

## Runner and launch

Reuse runs/overnight-streaming-perf/run.py (SHA256 2e55a19740f9244c7ad515f31c53c1be2a8f621aa36e7a71c42c51b82d99992f) and its measurement adapter (SHA256 8288698d1495cfdfed66953e22b0e84fc8b93f4720702f979309d38e9be99194). It sources /etc/profile.d/wm.sh, uses task-local caches and tmpdir, starts one eight-rank torchrun group, and cleans only that owned process group. The existing ignored output root is runs/overnight-streaming-perf/artifacts/raw. Per-run logs, resolved config, source snapshot, process status, summary, and completion paths are listed in pins.json.

Performance command (baseline control; use a unique run ID):

    /opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py --run-id <unique-run-id> --output-root "$PWD/runs/overnight-streaming-perf/artifacts/raw" --num-prefetch-batches 32 --epoch-size 51200 --steps 20 --warmup 10 --window 10

Trace command (capture update 11, at most three microbatches on ranks 0 and 1):

    /opt/torchtitan-venv/bin/python3.12 runs/overnight-streaming-perf/run.py --run-id <unique-run-id> --output-root "$PWD/runs/overnight-streaming-perf/artifacts/raw" --num-prefetch-batches 32 --epoch-size 51200 --steps 11 --warmup 10 --window 0 --profile-update 11 --profile-microbatches 3

The current runner does not yet define a fusion selector; use the task-owned selector once added. These commands are recorded, not executed by this inventory.
