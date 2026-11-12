# Frozen parameter storage

GIFT automatically stores frozen floating-point parameters in BF16 with both
FSDP2 and GFSDP2. `training.frozen_dtype` defaults to `"bfloat16"`, currently
the only supported value. Trainable parameters keep `training.dtype` (FP32 by default);
buffers keep their existing dtype.

Freeze during model construction using `FrozenConfig` or LoRA. Casting happens
before parallelization and allocation from meta; no extra configuration is needed.
GFSDP2 groups parameters by mesh, storage dtype and trainability. Compatible
data arenas are reused across units; simultaneously live groups use distinct
slots. Automatic resharding keeps at most the two final units unsharded when
their slots are disjoint, avoiding redundant backward all-gathers. These slots
can be released on demand for another forward or a different backward order;
explicit retention remains reserved. Frozen groups have no gradient or EMA buffers.
GFSDP2 builds shard metadata from parameter ranges and copies local slices directly,
without full-size GPU index arrays or a second full weight buffer.

FP32 checkpoints load into frozen BF16 parameters with rounding. Names stay the
same, but saved BF16 weights cannot recover the discarded FP32 precision.
Explicit FP32 computation using frozen weights may therefore change. Changing
`requires_grad` after GFSDP2 sharding remains unsupported.

Frozen weights use half the storage of FP32. Peak GPU memory savings depend on
simultaneously live partitions, prefetch and retained parameters.
Unused CUDA cache is released once after startup checkpoint loading; this does
not reduce the initialization peak.

Tests: `tests/unit_tests/test_frozen_parameter_storage.py` checks storage policy.
`test_gfsdp_initialization.py` checks packed layout, aliases and constructor allocations.
The models integration suite checks distributed numerics, checkpoint restore,
and Trainer save/resume with both backends.
