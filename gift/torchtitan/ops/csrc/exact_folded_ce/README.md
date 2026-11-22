# Exact folded cross-entropy

These sources reproduce the CUDA shared library used for the C2 MTP1 LM-head
acceptance runs. The GIFT runtime selects launcher method 13, which writes weighted
BF16 gradients into the logits allocation and reports FP32 row losses. It supports
the validated H100 input with local batch 2, sequence length 4096, and vocabulary
128256. The original source retains the other launcher methods to preserve the
validated build identity.

Build from this directory using NVIDIA CUDA 13.2.51:

```sh
nvcc -O3 -shared -Xcompiler=-fPIC -lineinfo -Xptxas=-v -arch=sm_90a \
  --fmad=false --cudart=static centered.cu -o centered.so
shasum -a 256 centered.cu exact.cuh centered.so
```

| File | SHA-256 |
| --- | --- |
| `centered.cu` | `ab4ce470a5a9eadb41bcff4e22f0a9470d6772465cba5011c323f5ab3e16136c` |
| `exact.cuh` | `32c94dac3d213c236506854b69f24da2defb97a1e9275cf976aa1947e1bc2b6f` |
| `centered.so` | `41357457bb454c721ee72b5b9e4151cff7056f5e3b473a1241b39c4161925a11` |

Set `GIFT_EXACT_FOLDED_CE_LIBRARY` to the absolute path of the validated
`centered.so`. The runtime requires the recorded digest. A rebuilt binary with a
different digest needs validation before changing the pin.

The exact kernel requires `GIFT_STABLE_CE_FOLDED_BACKWARD=1`,
`GIFT_EXPERIMENT_LM_HEAD_MATRIX=new`, and
`GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS=2` with Inductor loss compilation,
`loss.num_chunks=2`, and one MTP head. The accepted recipe also sets
`GIFT_DIRECT_BF16_DHIDDEN=1` and `GIFT_FUSED_HIDDEN_Z_LOSS=1`, and uses
regular BF16 GFSDP all-gather.
