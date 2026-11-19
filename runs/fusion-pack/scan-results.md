# FlashQLA direct backward on the native GDN scan boundary

`GIFT_FLASH_QLA_DIRECT_BACKWARD=1` removed one output scan from the stateless `external_gate` backward. On the native rank-zero B2/S4096 fixture, the output and all seven gradients matched bitwise in eager execution and under `torch.compile(fullgraph=True)` with the production selective checkpoint policy. All 26 focused backend tests passed. [Measurements and checks](scan-results.json)

The pinned FlashQLA backward consumes FP32 cumulative gates, the KKT result and the incoming gradient. It recomputes recurrent chunk states, runs the pinned fused backward, reduces grouped query/key heads and reverses the cumsum. The direct VJP invokes the same FLA gate forward and backward and the same required FlashQLA kernels. The outer `torchtitan::gdn_chunk` and `torchtitan::fla_vjp` interfaces and checkpoint policy are unchanged. CP and recurrent state are outside this backend's supported contract. [Pinned FlashQLA source](https://github.com/QwenLM/FlashQLA/blob/a14dbf3474514e1674526431e960325f72081e9f/flash_qla/ops/gated_delta_rule/chunk/__init__.py) · [Pinned FLA snapshot](../../gift/contrib/flash-linear-attention/fla/ops/gated_delta_rule/gate.py) · [Implementation](../../gift/torchtitan/experiments/flash_qla/backend.py)

## Latency and peak memory

Each scope used baseline, direct, direct, baseline order on physical GPU 1, an H100 80 GB HBM3 on `sr008-gpu1`. The table shows the median of 24 samples per implementation from two 12-sample blocks. Device graph samples average 20 warm-cache replays on immutable fixture storage after five warmup calls. The backward-only scope excludes its preparatory forward. The compiled scope includes the checkpointed GDN call and its backward, including any scoped SAC work. The separate synchronized calls include Python, autograd, dispatch, GPU work and the wait on the trailing event. The traced launches are from one call per implementation and scope. Latency percentages use the corresponding baseline median.

| Scope | Metric | Baseline | Direct | Change |
| --- | --- | ---: | ---: | ---: |
| Backward only | CUDA Graph replay, ms | 1.148 | 0.976 | −14.99% |
| Backward only | Synchronized call, ms | 1.603 | 1.222 | −23.77% |
| Backward only | CUDA event elapsed, ms | 1.598 | 1.215 | −23.97% |
| Backward only | Host call, ms | 1.249 | 0.634 | −49.21% |
| Backward only | Trailing event record and wait, ms | 0.343 | 0.587 | +70.87% |
| Compiled SAC | CUDA Graph replay, ms | 1.360 | 1.220 | −10.28% |
| Compiled SAC | Synchronized call, ms | 2.038 | 1.778 | −12.74% |
| Compiled SAC | CUDA event elapsed, ms | 2.030 | 1.771 | −12.78% |
| Compiled SAC | Host call, ms | 1.566 | 1.202 | −23.25% |
| Compiled SAC | Trailing event record and wait, ms | 0.471 | 0.573 | +21.70% |

The backward-only baseline block medians for device replay were 1.148 and 1.130 ms, a 0.018 ms spread. The compiled SAC baseline block medians were 1.360 and 1.359 ms. The earlier host return leaves more outstanding work for the trailing event in both measurements. These asynchronous durations overlap, and the individual medians use their own sample distributions.

Peak allocated memory fell from 707.50 to 643.50 MiB (−9.05%), and peak reserved memory fell from 730 to 666 MiB (−8.77%), in both scopes. The 64 MiB difference matches the BF16 output size at `[2, 4096, 32, 128]`. After a cache clear, the backward-only case starts at 257.00 MiB allocated including the prepared forward, and the compiled SAC case starts at 193.00 MiB. Allocation measurements exclude CUDA Graph capture. These are single-GPU component peaks.

## Trace verification

| Scope | Output-scan launches | All compute kernels | Kernel duration sum, µs | Device span, µs | Device gaps, µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Backward only | 1 → 0 | 15 → 14 | 1,154.14 → 994.93 | 1,746.40 → 1,131.92 | 591.55 → 136.25 |
| Compiled SAC | 2 → 1 | 19 → 18 | 1,355.63 → 1,212.34 | 2,060.91 → 1,809.56 | 704.54 → 596.48 |

All remaining kernel names, grids and block sizes matched within each pair. The removed output scan took 146.75 µs in the backward-only trace and 142.56 µs in the compiled SAC trace. The compiled candidate retained an output scan of 142.62 µs. Each trace has one memset under 1 µs and zero copy or collective events. These traces explain the mechanism. The unprofiled measurements above estimate latency.

## Provenance and reproduction

The fixture captures the first rank-zero GDN backward from the fixed production-tokenizer `data-002` run `fusion-pack-scan-acquire-002`. The native run passed on all eight ranks. Resolved config SHA-256 is `ba0a1d8183f9c100e6a4cbea59ff9d04722052afcb32f79cf9c244bb8244d309`. Fixture SHA-256 is `78feea4706f14091966df48c6b91bde6208745fba0189bda0cd59018b9aee67c`. The replay preserves shapes, strides, BF16 inputs and incoming gradient, with storage offsets rebased. The compiled and eager comparisons use identical operands. PyTorch was `2.15.0.dev20260822+cu130`, CUDA 13.0, Triton 3.8.0 and TileLang 0.1.12. [Capture adapter](scan_capture.py) · [Baseline workload](../../docs/baseline.md#current-final-task-baseline)

The replay ran at revision `fff35b0b8d6c0d98ca59b4bf58f6c326333ea045` from the SR008 task root with only the assigned GPU exposed.

```bash
scan_replay_run_id="scan-component-replay-$(date -u +%Y%m%dT%H%M%SZ)"
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$PWD/gift" \
  /opt/torchtitan-venv/bin/python3.12 -Bu runs/fusion-pack/scan_component.py \
  --fixture runs/kernel-fusion/artifacts/raw/fusion-pack-scan-acquire-002/component-inputs/fixture-01.pt \
  --output "runs/fusion-pack/artifacts/raw/$scan_replay_run_id/replay"
```

`scan-component-001` exited after the parity checks and the first baseline trace because autograd retained a default-stream leaf during CUDA Graph capture. The rerun used fresh leaf identities on the capture stream with the same fixture storage. `scan-component-002` passed, and the remote `scan.json` SHA-256 matches the saved [results](scan-results.json). The component evidence covers the observed dense shape and CP1. Training throughput, full-step memory, and interactions with the other candidates require the separate native experiment.
