# Loss-only matched performance matrix

Prepared and statically reviewed; no launch release yet.

Run C0a/C1a/C0b/C1b serially on frozen source006. Both variants exclude all18 model blocks from graphs while retaining their compilation and selective checkpointing, FlexAttention, FP8, HybridEP, original batches/sequence and accumulation4. The global graph flag is the sole intended variant difference. Common mixed-order reduction is off; graph partitioning remains on. The A008 independent review proves persistent real loss forward/backward replay; its native execution/log validity also passed.

Each native process stops at71 with scheduler horizon30000 and seed42. Measure updates21–70, closed at71:52,428,800 actual already-global tokens divided by the slowest completed window. C0a/C0b provide A/A noise. No profiler, extra compiler logs, checkpoint or evaluator is enabled. Reuse the existing observer/runner/cache policy. Exact scope, checks, hashes and all four prepared attempts are in `protocol.json`.

Root must verify each fresh same-eight-H100 idle/app-empty and storage write preflight and release each attempt individually. A new logging persistence failure or incomplete control stops the matrix; no retries. Preserve C1's earlier ENOSPC and event-loss evidence. These measurements do not establish numerical acceptance or checkpoint continuation. Allocator boundary/native interval and sampled device memory are available; exact all-rank phase allocator peaks remain a named gate for separate resolution before final acceptance.
