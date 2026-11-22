# Routed BF16 weight-gradient overwrite

This extension handles supported Hopper grouped FP8 projections when the
backward pass has no direct expert dW slot. The kernel accumulates in FP32,
rounds to BF16, adds positive zero in registers, and writes the BF16 output.
The binding zeros the skipped slices for experts with no tokens. The patch and
binding are the frozen inputs from the accepted six-fix Hopper run.

Build in the matching training image with CUDA 13.0 and installed DeepGEMM
`2.3.0+f093a4c`. Set `DEEPGEMM_DONOR` to the flattened DeepGEMM
`f093a4cd643bc06a7894a90dee852ec70e989a53` source, including its pinned
CUTLASS and fmt trees. Set `BUILD_ROOT` to a new absolute path in your
task-owned area. The build verifies the donor tree, installed DeepGEMM,
patch, binding and staged source. [Pinned DeepGEMM source](https://gitlab.ai.cloud.ru/gigallms/deepgemm/-/tree/f093a4cd643bc06a7894a90dee852ec70e989a53)

From the final-all-fix checkout, using the training interpreter:

```bash
CUDA_HOME=/usr/local/cuda MAX_JOBS=1 python -B \
  runs/final-all-fix/routed-dw/build.py \
  --donor "$DEEPGEMM_DONOR" --build-root "$BUILD_ROOT"
```

The build writes `manifest.json`, `extension/shk_routed_dw_bf16.so`, and the
staged `source/` tree. Keep them together for the runtime checksum checks
and kernel JIT compilation. For validation of source staging without the
training image, add `--stage-only` and use a separate build root.

Set `GIFT_ROUTED_DW_BF16_OVERWRITE=1` and
`GIFT_ROUTED_DW_BF16_OVERWRITE_BUILD="$BUILD_ROOT"` to make this build
available to eligible grouped backward calls. The final-all-fix smoke runner
pins the original build receipt for the recorded campaign. A local rebuild
produces its own binary and receipt hashes.
