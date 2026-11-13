# Stable CE: representative local numerical replay

The recorded CE block passes a precision-aware comparison against V2. This is
a synthetic local forward/backward check, not a convergence or whole-model
continuation result. [Compact results](ce-numerics.json) own measurements and
runtime/source pins; the original JSON and log remain under ignored raw artifacts.

## Matched path

The [replay script](ce-numerical-replay.py) calls the actual native and stable
CE functions through `torch.compile(..., backend="inductor")`, preserving native
compile defaults. Both use the identical BF16 logits storage, int64 labels,
Python-float denominator 2,096,640, and `(raw_sum / denominator) * head_weight`
before unit-upstream backward. Head weights are 1.0 and 0.3.

Seed 2048 produces synthetic standard-normal BF16 logits. Each of the two outer
chunks has shape `[4, 2048, 128256]`: main batch followed by MTP batch. Labels
shift over the full 4096-token sequence before chunking, retaining the boundary
target and final ignored positions. This isolates the changed CE block; head
projection, distributed communication and optimizer state are outside the replay.

## Results and interpretation

| Measurement | Chunk 1 | Chunk 2 |
| --- | ---: | ---: |
| Compared BF16 gradient elements | 1,050,673,152 | 1,050,673,152 |
| Bitwise-equal gradient elements | 1,050,609,442 | 1,050,602,899 |
| One-ULP differences | 63,710 | 70,253 |
| Differences greater than one BF16 ULP | 0 | 0 |
| Maximum absolute gradient difference | 4.5475e-13 | 4.5475e-13 |
| Gradient relative L2 difference | 1.7546e-7 | 1.7231e-7 |
| Maximum raw head-loss difference | 0.00390625 | 0.00390625 |
| Weighted chunk-loss difference | 1.8626e-9 | 0 |

The raw loss difference is one FP32 ULP at the observed roughly 50,000 sums.
The total weighted loss is exactly the same recorded FP32 value,
`0.06219938397407532`; no general bitwise-equality claim follows. FP32 incoming
scales match bitwise for both heads (`0x35000801`, `0x3419a335`). Shapes and
dtypes match, every gradient is finite, and ignored-label gradients are exactly
zero across both full chunks.

FP64 centered-softmax references cover 20 and 22 rows, including spread,
head/batch boundaries, ignored rows and the worst changed rows. They use the
represented BF16 inputs and the actual incoming FP32 scales. Both implementations
are at most one BF16 ULP from the ideally rounded reference on those rows.
Reference relative L2 errors are approximately 6.59e-4 and 6.48e-4 for both;
maximum absolute errors are identical within each chunk. The differences fit
the inspected FP32 recurrence, exponential and reduction-order changes followed
by BF16 rounding. There is no unexplained material difference in this case.
The prior large-common-offset regression remains separate evidence.

## Execution and checks

After the corrected trace stage stopped and all eight GPUs were verified idle:

```bash
source /etc/profile.d/wm.sh
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 /opt/torchtitan-venv/bin/python3.12 \
  runs/four-fusion/ce-numerical-replay.py \
  --output runs/four-fusion/artifacts/raw/tasktmp/ce-numerical-replay-001.json
```

The process exited zero, completed both chunks and released GPU 2 before the
unprofiled repeat matrix began. Script AST, normal Ruff and format checks passed.
An initial shell launch failed before starting Python because its log directory
did not exist; creating the task-owned directory resolved it. No GPU work
occurred in that failed launch. No training source/configuration was changed.
