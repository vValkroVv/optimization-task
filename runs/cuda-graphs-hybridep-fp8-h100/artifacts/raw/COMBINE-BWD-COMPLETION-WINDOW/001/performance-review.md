# Pre-edit performance review: combine backward completion window

Source context: task-local `gift` diagnostic at base revision
`446431dfa191ef0300f116257ae918bc85e38c06`. The only planned source edits are
the completion observer, its focused unit tests, and its maintained diagnostic
README section.

The callback is opt-in and paired with the dedicated
`train_combine_bwd_completion` entrypoint. The ordinary `train` and
`train_scan_fix` entrypoints do not install its adapter. The supported target
keeps the 3B MoE/HybridEP EP8/FSDP2 preset, selective AC, model/loss Inductor
compilation, FP8, four microbatches per update, and graph-off setting. The
adapter's selected-call path invokes the existing helper once, then records and
waits on one no-timing CUDA event and flushes enter/complete records for each
active selected call.

Extending the fixed active interval from updates 21–40 to 21–60 leaves the
first 20 updates inactive and adds 20 active updates, or 80 active microbatches
per rank at four microbatches per update. The actual selected-call count per
microbatch remains workload-dependent; the change does not assume a fixed layer
count. On the opt-in path, the added interval repeats existing Python selector
and counter work, event creation/record/wait, and flushed JSONL writes in
proportion to actual selected calls. It adds no tensor computation, tensor
copies/conversions, collectives, or new synchronization type. Observer state
retains 20 more update boundaries and their per-microbatch counts; log volume
and event objects scale with observed calls and remain unmeasured.

The event wait reduces overlap and can perturb allocator reuse and temporary
object lifetime, so the diagnostic may mask a timing/lifetime fault. Its runtime
cost and effect on numerical behavior require the authorized GPU attempt and
remain unknown here. This bounded change does not alter normal-entrypoint
runtime, the production preset, precision, AC, compilation, graph exclusions,
communication, or training arithmetic. It is a diagnostic extension, not a
performance optimization or repair. No GPU, benchmark, installation, or
shared-resource action is authorized in this implementation step.
