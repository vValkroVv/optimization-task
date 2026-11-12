# Checkpoint retention changed-state receipt

**Status:** stopped; deletion was not performed by this worker.

**Target:** `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/attempts/001/create_s0/checkpoint/step-1`

The existing metadata audit at `runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/NUMERICAL-NATIVE-PROBE/analysis/001/extraction.json` recorded nine files totaling **39,211,598,683 bytes (36.518647 GiB)**. Its recorded `.metadata` SHA-256 is `d37d0fd6f0d04ec43a70daafdb86f76a8d70ca39eb921cabc539552d674f1784`.

| File | Previously observed size (bytes) |
|---|---:|
| `.metadata` | 2,145,827 |
| `__0_0.distcp` | 4,900,700,072 |
| `__1_0.distcp` | 4,901,251,523 |
| `__2_0.distcp` | 4,901,247,096 |
| `__3_0.distcp` | 4,901,247,352 |
| `__4_0.distcp` | 4,901,244,367 |
| `__5_0.distcp` | 4,901,255,898 |
| `__6_0.distcp` | 4,901,256,026 |
| `__7_0.distcp` | 4,901,250,522 |

On 2026-11-12, the worker resolved the target inside the task repository, observed nine regular non-symlink files matching the audit, and completed a streaming SHA-256 pass over all nine (8 MiB reads; payload values were not interpreted). The pass reported the audited total and the `.metadata` digest above. Per-file digest lines were emitted in the tool output, but no durable inventory was written before the target changed. A manually carried digest for `__1_0.distcp` failed a 64-character guard, so this receipt does not claim a complete, verified per-file hash inventory.

The attempted inventory script stopped before creating the evidence directory or deleting anything. A subsequent rehash of `__1_0.distcp` failed with `FileNotFoundError`; the next read-only check observed `step-1` absent and the evidence destination absent. No unlink was issued by this worker or the root agent. The cause and any external actor are **unknown**. The metadata copy was not made. No authorized deletion completion is claimed.

Earlier process inspection found no open descriptors to the target files; the only matching command during the check was the worker's own metadata `sha256sum` invocation. `lsof` was unavailable. This receipt records the observed status and is not a recovered pre-deletion inventory.

**Hash output provenance:** the following values are copied from the worker's streamed-hash command output before disappearance; they are retained as observed output, not as a saved manifest or post-disappearance verification. The carried `__1_0.distcp` string failed the worker's 64-hex-character guard and is intentionally omitted.

| File | SHA-256 copied from command output |
|---|---|
| `.metadata` | `d37d0fd6f0d04ec43a70daafdb86f76a8d70ca39eb921cabc539552d674f1784` |
| `__0_0.distcp` | `45d7d57e8724d4bd196e4c7ee619382e8119883321788f0f38caa600cc825610` |
| `__1_0.distcp` | not retained; carried string failed the 64-character guard |
| `__2_0.distcp` | `67f4042ae23942092c796cfdf904ef9fe17719acdfa80e7ee06c47ae9d571349` |
| `__3_0.distcp` | `cae0bc95afb077762b2c5fa22523d01c69856dc467f60a34ec46f54b088cc2a8` |
| `__4_0.distcp` | `93282b7d739bc75017a1f05484087dfe0d1cdfe6c996dca62163cabda93c9af5` |
| `__5_0.distcp` | `c110817d5d98bd0c8aede128196ab4fc236b3e8bfd4e8ae2ca56880a80134b16` |
| `__6_0.distcp` | `dcacde3e3e3f1a70fe424a7041ce448b7a0af264cc78d946b17da7b55c4015cc` |
| `__7_0.distcp` | `9140ea35d5ab11e3da536faf2572fd865dd504523e3bd10b2a44c8517dfe9600` |
