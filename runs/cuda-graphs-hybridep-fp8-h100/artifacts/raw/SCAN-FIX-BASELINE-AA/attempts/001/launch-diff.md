# Launch/config difference from prior corrected smoke

- Entry module remains `torchtitan.experiments.cuda_graph_campaign.train_scan_fix`; no custom training loop.
- Original graph policy remains `TORCHINDUCTOR_CUDAGRAPHS=0`.
- Native stop changes from 25 to 71 updates.
- Observer overlay changes from updates 3–7 (five-update diagnostic) to updates 21–70 (50 updates), closing at the executed update-71 boundary.
- Scheduler horizon stays 30,000; seed stays 42; checkpointing stays disabled; the exact preset stays at SHA-256 `b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b`.
- Attempt-local runtime/dump/log roots differ. Existing offline data/Inductor/Triton/DeepGEMM caches are reused; no cache copies or CP/tensor dumps are created.
- The prior smoke's mode, FP8, selective AC, model/loss compile, fused AdamW, c4_test, batch/sequence shape, and EP8/FSDP2 topology remain supplied by the unchanged preset.
