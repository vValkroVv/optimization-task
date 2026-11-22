# Final all-fix recipe

The EP8 and EP4 YAML files import the compiled Original A C2 crop at
`e9b7cdbb4`. Both use the native FLA scan and five FLA/MLA changes, a depth-32
CPU prefetch queue, the GPU capacity selector in both MoE block families, and
the `dg-n4c` four-channel compact row-owner embedding with the layer-0 and MTP FFN
splits and MTP transition. Both use exact consuming CE, direct BF16 hidden
gradients, direct FP32 head weight gradients, fused hidden z-loss and regular
BF16 head weight gather.

Expert FP8 caching, direct expert dW with deferred waits, and tail retention
are enabled on both meshes. EP8 uses the existing FP32 direct slot. EP4
exchanges compact BF16 contributions and accumulates on each FP32 shard
owner. The SM90 routed BF16 overwrite is available for a returned gradient
when a group has no direct slot. The accepted direct writer owns all seven
expert groups in both recipes. The other five SM90 optimizations operate
on dense weight tails and CuTe forward, replay and backward.

## Launch

Use the personal SR008 checkout, the installed eight-H100 job runtime and the
same verified build artifacts as the group comparisons. From the branch root:

```bash
source /etc/profile.d/wm.sh
task=/workspace-SR008.nfs2/vkropotin/optimization-task
direct_dw_build="$task/artifacts/dw-Zi0UA8/build"
ce_library="$task/runs/head-frontier-nov/fused-ce/centered.so"
overwrite_build="/workspace-SR008.nfs2/vkropotin/shk-fix/runs/shk-fix/artifacts/raw/routed-dw-bf16-001/build"
short_personal_scratch="$task/.tmp"
run_id="final-ep8-$(date +%Y%m%d-%H%M%S)"
/opt/torchtitan-venv/bin/python3.12 -Bu runs/final-all-fix/run.py \
  --ep 8 --run-id "${run_id}" --probe --steps 1 \
  --direct-dw-build "${direct_dw_build}" --ce-library "${ce_library}" \
  --overwrite-build "${overwrite_build}" --scratch-root "${short_personal_scratch}"
```

Select `--ep 4` for eFSDP2. The launcher verifies the pinned libraries, data
and tokenizer against their receipts, requires eight idle H100s, and uses
separate caches under each attempt. The short writable scratch path holds
multiprocessing sockets. The probe records CPU/CUDA execution for one
microbatch per rank and completes all 32 microbatches and the optimizer step.
For an ordinary run, omit `--probe` and set `--steps` to the intended length.

Logs, resolved config, and per-rank receipts are under
`runs/final-all-fix/artifacts/raw/<run-id>/`. Verify a completed probe with:

```bash
/opt/torchtitan-venv/bin/python3.12 runs/final-all-fix/verify.py \
  "runs/final-all-fix/artifacts/raw/${run_id}" --ep 8
```

## SR008 activation checks

| Attempt | H100 ranks | Microbatches per rank | Loss | Gradient norm |
| --- | ---: | ---: | ---: | ---: |
| `final-ep8-002` | 8 | 32 | 15.32916 | 1.0904 |
| `final-ep4-002` | 8 | 32 | 15.33188 | 1.0906 |

Both runs finished one optimizer step. On the sampled microbatch, every rank
executed 14 GPU capacity-mask kernels and 14 direct expert weight-gradient
writes. EP8 used the rounded FP32-slot kernel. EP4 used the compact BF16 slot
and executed 14 owner writes. The per-rank ranges for FLA, MLA, CuTe, head
and dense weight quantization are in [verification.json](verification.json).

The profiler affects throughput and memory. Use matched unprofiled 10+10
comparisons on each topology for performance claims.

On both runs, Python's multiprocessing finalizer reported `EBUSY` while
removing temporary NFS files during shutdown. All eight rank receipts were
written and torchrun returned zero for both attempts.

The separate CPU/Gloo checkpoint-continuation test fails on the first
unsplit control backward, before writing a checkpoint. Both CPU ranks report
a BF16 tensor version mismatch on the SR008 training interpreter. The same
failure reproduced on macOS with the isolated `dg-n4c` changes. The GPU
activation probes cover training startup and one update. Checkpoint resume
remains unverified. The CPU log is in `artifacts/raw/verify-dcp-cpu.3jo2b8/`.
