# FlashQLA raw-gate experiment

`install_backend(variant)` explicitly installs a strict priority-zero FLA GDN
backend. Importing the module does not register a backend. Supported variants
are `fla`, `external_gate`, and `fused_gate`; selection also overrides the
builtin FlashQLA backend for the explicit `fla` reference. Unsupported inputs
raise in the selected implementation rather than permitting dispatcher fallback.

The integration preserves the existing GDN opaque op, selective activation
checkpointing, and recompute VJP. Inputs have BF16 q/k heads 16x128, value heads
32x128, raw gate and beta; A_log/dt_bias may be BF16 or FP32. Gate arithmetic and
cumulative gates are FP32. Context parallelism and recurrent states are excluded.
The default scale is 128**-0.5. Q/k normalization occurs upstream.

The external variant computes the FLA gate and invokes the public FlashQLA
chunk operation with automatic CP disabled. The fused variant replaces only
FlashQLA's gate/cumsum forward orchestration using FLA's existing fused gate
cumsum. It reuses the pinned FlashQLA backward, which already reverses the
cumsum and reduces GVA gradients, then applies the FLA gate backward.
Upstream revisions and actual source paths/hashes are returned by installation;
the task HEAD is recorded separately from the upstream pins.

## Runtime costs and validation limits

The external variant retains a separate FP32 gate intermediate and gate kernel.
The fused variant retains cumulative FP32 gates and KKT output for autograd,
avoiding the separate per-token gate output. Backward still recomputes FlashQLA
states; the native opaque-op VJP still repeats the forward. Checkpoint replay
therefore retains the native 2F+B or 3F+B execution pattern. No collectives,
device scalar reads, or new device synchronization are added. Activation logging
occurs once per variant and grad mode; source validation and kernel module
loading are cached, with no repeated filesystem I/O in the hot path.

CPU tests cover guards, real FLA registry selection/failure behavior, and custom
autograd orchestration with kernel doubles. They do not validate GPU arithmetic,
TileLang compilation, GVA kernel behavior, peak memory, CUDA Graphs, compiled
training, checkpoint replay, or performance. These require the authorized GPU
validation harness before production acceptance. No speedup is claimed.
