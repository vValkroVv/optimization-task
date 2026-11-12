# Scan-fix deterministic A/A handoff

## Control and command

Two independent, fresh eight-rank processes used the same frozen source snapshot, SOURCE-SNAPSHOT/002, tracked patch SHA-256 b7c02e66f19d7ca2558960660dee91e9771b2d06b75a1556416838e27ed05465, and original preset gift/configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml, SHA-256 b341701bd1eef7774c110fbf7ed69cf6280ba8691100b05be457dee3e7d50a7b. No GIFT source changed between these runs.

Each attempt used this same command, changing only its output and scan-runtime directories:

    /opt/torchtitan-venv/bin/python3.12 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 --log-dir=$ATTEMPT_DIR/torchrun --redirects=3 --tee=3 -m torchtitan.experiments.cuda_graph_campaign.train_scan_fix configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml --training.steps 10 --lr_scheduler.total_steps 30000 --debug.seed 42 --debug.deterministic --debug.deterministic_warn_only false --checkpoint.enable false --metrics.enable_tensorboard true --metrics.log_freq 1 --dump_folder $ATTEMPT_DIR/native

The launcher prepended task-local gift to PYTHONPATH, kept HOME unchanged and offline, reused the existing TorchInductor/Triton and DeepGEMM caches, and set TORCHINDUCTOR_CUDAGRAPHS=0. Each process had its own GIFT_HYBRID_EP_SCAN_RUNTIME_DIR and fresh per-rank DeepEP native JIT cache.

## Result and strict coverage

Both attempts exited 0 and resolved to 10 updates, scheduler horizon 30000, seed 42, deterministic algorithms enabled with warn_only=false, checkpointing disabled, TensorBoard enabled at every step, and model/loss compilation enabled. All eight ranks recorded 10 optim_end and 10 step_end structured events. Both runs persisted steps 1–10 for loss/train/total and l2_norm/grad/global. All-rank runtime manifests record the same corrected HybridEP header SHA-256, ff3a22e1c4e15785d9beca011a457f138dd1ce6502c7240ce6432d0f0985daba.

The strict comparison uses exact persisted TensorBoard Summary.Value.simple_value float32 bit patterns, with no tolerance. It FAILS: both required tags match at step 1 and mismatch at steps 2–10 (9 of 10 updates). The scan-bound correction therefore does not restore strict loss/norm A/A. The gate remains failed.

Evidence:

- 001 result and metrics: 001/result.json and 001/metrics.json
- 002 result and metrics: 002/result.json and 002/metrics.json
- Strict pair comparison: comparison.json
- Frozen source identity: source-identity.json
- GPU ownership handoff: gpu-ownership-handoff.txt

## Extractor subattempt

The first all-rank activation-marker assertion failed with the exact message: Expected 8 activation markers; saw 1. That extractor searched only the combined supervisor.log for one INFO marker per rank. GIFT logging emits that INFO message only on rank 0, so this was an analysis assertion, not a training or GPU failure.

The corrected validation uses each rank’s scan-runtime/rank-*/runtime/runtime_manifest.json for staged/source header hashes and unique native-cache paths, and each rank’s structured JSONL optim_end/step_end records for optimizer-step completion. The marker visibility and correction are recorded in 001/analysis-subattempts/001-extraction-assumptions.json. No training run was repeated for this assertion.
