# Reused LM-head projection buffer

## Result

A per-loss BF16 projection workspace was tested at 8 and 16 fixed outer chunks on physical H100 GPU 0. The existing frontier_stack.py stack command was the unchanged control. The workspace arm changed only _StagedHeadLinear.forward in the benchmark process.

The stack harness produced exactly matching SHA-256 digests for loss, BF16 dHidden, FP32 dW, z-loss, and z-loss gradient at both chunk counts. The workspace gate passed on every reuse: 23 per-loss allocations and 161 ordered reuses at 8 chunks; 23 allocations and 345 ordered reuses at 16 chunks. All tested activation markers were present, the tuned cuBLASLt dW writer ran, and both arms recorded zero allocator retries and OOMs.

| Chunks | Control median (ms) | Workspace median (ms) | Difference (ms) | Peak total GPU (GiB), both arms |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 78.512 | 77.769 | -0.742 | 10.280151 |
| 16 | 89.566 | 89.290 | -0.276 | 10.280151 |

Each median is over the stack harness's 10 synchronized host-wall samples in one process. The samples show drift and outliers, so these small median differences do not establish a speedup. The peak-memory metric is the harness's peak allocator-reserved bytes plus non-PyTorch device memory observed after each sample. It was identical across arms and chunk counts, so the repeat condition (meaningful memory gain with no material timing cost) was not met. No repeat was run.

The integrated timer includes host dispatch, synchronization, label packing, head/CE forward and backward, BF16 dHidden, FP32 GFSDP dW, and the independent hidden z-loss. Device compute, host submission, synchronization waits, and transfers were not attributed separately. This is a single-rank replay, not a distributed-training result.

## Workspace behavior

The wrapper preserves the native saved input and GFSDP weight and stores the actual staged dW lease on ctx. It lazily allocates one BF16 [flattened rows, vocabulary] buffer on that lease, calls torch.mm(input.flatten(0, -2), weight.T, out=buffer), and returns the native logical output shape. Before reuse it checks the same CUDA stream and that the previous head backward has advanced both lease writes and completed callbacks. The existing native backward remains unchanged.

For the captured fixture (vocabulary 128256), each workspace is about 0.489 GiB at 8 chunks and 0.245 GiB at 16 chunks. This benchmark did not reduce total peak GPU memory versus the stack control.

## Reproduction and evidence

- Worktree revision: 2c67c182e8de6145f3796d62203714ab1b815446
- Hardware: NVIDIA H100 80GB HBM3, physical GPU 0, UUID GPU-8f5a8eb7-d9b2-6c7d-b377-5953bcf6c246; driver 570.133.20
- Runtime: PyTorch 2.15.0.dev20260822+cu130, CUDA 13.0
- Run order: 8 chunks control ºw^~)Þv workspace; 16 chunks workspace ºw^~)Þv control. Runs completed 2026-11-20 18:14:07ºw^~)Þw18:15:22 UTC.
- Activation: stable folded CE, direct BF16 dHidden, staged head dW, and fused hidden z-loss were all 1; dW used the preserved cuBLASLt tuning plan.
- GPU 0 was idle before each process; after the lane completed, nvidia-smi reported 0 MiB and no compute-app process.

Commands from the authorized worktree:

    bash runs/lm-head-mtp/frontier_workspace_lane.sh 0 8 initial control workspace
    bash runs/lm-head-mtp/frontier_workspace_lane.sh 0 16 initial workspace control

The benchmark-only wrapper is [frontier_workspace.py](frontier_workspace.py); the runner is [frontier_workspace_lane.sh](frontier_workspace_lane.sh). Raw timing, memory, activation, writer, and digest receipts are in [chunks-8](../../artifacts/frontier-workspace/initial/chunks-8) and [chunks-16](../../artifacts/frontier-workspace/initial/chunks-16). Training source was not modified.
