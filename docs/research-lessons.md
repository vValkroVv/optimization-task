# Research lessons and candidate hypotheses

These lessons guide investigation; they are not results for the selected pretraining workload. The SFT evidence does not establish pretraining quality, speed, or support for another model, topology, sequence length, or source revision. Pin and inspect the implementation before adapting any idea.

## Lessons that constrain interpretation

- Sequence shape, packing, padding, and supervision affect physical work, targets, and boundaries separately. A loss mask does not remove model or expert-dispatch work. Preserve isolation across attention and recurrent convolution/scan state.
- A small attention fixture can establish a route or correctness property but cannot prove long-context memory fit, whole-model capacity, or end-to-end performance. Backend availability does not prove the backend is active.
- FSDP resharding, prefetch, and buffer reuse trade communication, overlap, and live memory. Their direction depends on groups, topology, module sizes, and schedule.
- Compilation and activation checkpointing affect rounding, buffer lifetimes, and side effects. Eager/compiled agreement and compile-state continuation need their own evidence.
- Resume correctness depends on loader position, RNG and optimizer state as well as model weights. A weights-only load is a different lifecycle.
- Pipeline balance depends on expensive heads, experts, embeddings, and schedule dependencies, not layer count alone. Communication bytes do not measure exposed waiting or integrated time.
- Historical SFT outcomes do not transfer to pretraining. A change in data distribution or objective requires workload-specific quality evidence.

## Candidate backlog

All entries below are hypotheses. Profile first, then select one candidate that matches a measured bottleneck and inspect a pinned implementation. The [initial task boundary](task-boundary.md) currently deprioritizes PP; retain it as a future profile-driven area.

| Candidate area | Question to resolve before changing code |
| --- | --- |
| Data pipeline, synchronization, and cadence | Is the GPU waiting on input, metric reads, validation, or checkpoint work at production frequency? |
| Padding and packing | Is avoidable physical work material, and can boundaries, order, objective, and exact counts remain valid? |
| Attention backend | Which backend actually runs at representative shapes, and does it fit memory with the complete live model state? |
| FSDP resharding and prefetch | Which groups and buffers lie on the critical path, and what is the paired live-memory cost? |
| Pipeline balance | Which stage or dependency limits throughput after accounting for heads, experts, and backward scheduling? |
| Compilation | Which graph or specialization is active, and do startup costs, numerical behavior, and state lifetimes fit the production run? |
| Expert-parallel overlap | Are dispatch/combine waits exposed, and can an overlap change preserve collective ordering and state? |
| Vocabulary-head chunking | Is head/loss storage or compute limiting the update, and are reduction order and loss semantics preserved? |

FP8 is required in the intended target baseline. Resolve its recipe and quality criteria before treating a specific setup as selected; preserve that recipe in optimization candidates. Changes beyond it need explicit agreement and quality validation.

## Provenance

Workspace provenance (these source files are adjacent to this repository in the authoring workspace and are not copied into this clone):

- `general-research/sft-lessons/README.md`: “The connections that matter” and “Reading map.”
- `general-research/sft-lessons/system.md`: “A row is a container; a target is a prediction,” “The vocabulary head, future heads and objective weights,” and “Backward completes the connection.”
- `general-research/sft-lessons/parallelism.md`: “DP and FSDP: examples versus storage,” “PP: stages, microbatches and backward dependencies,” and “The interactions determine the real cost.”
- `general-research/sft-lessons/workflow.md`: “The input pipeline has its own state,” “Activation checkpointing replays computation,” “Validation borrows the training process,” and “Persistence, caches and exact continuation.”
- `general-research/sft-lessons/findings.md`: “The mathematical function and its execution,” “Precision describes several boundaries,” “Accumulation and distributed reduction interact,” “Clipping connects all parameters,” and “Adam makes small gradients consequential.”
- `general-research/sft-lessons/performance.md`: “Shapes connect useful work to physical cost,” “Memory savings depend on lifetimes,” “Communication volume is only part of latency,” “Pipeline balance is more than layer count,” “Compilation and FP8 move several costs together,” and “Measure the recurring workload and the whole run.”

Additional workspace context: `sft-task/AGENTS.md` sections “Verification and evidence” and “Training-source code quality”; `sft-task/README.md` “Repository map”; and `sft-task/planning/task-workflow.md` “From evidence to the next action” and “Verification.” The lesson files link to implementation-specific documents and run reports under `sft-task/docs/` and `sft-task/runs/`; inspect those linked sources and their revisions before reusing any implementation claim. These are provenance pointers, not links available in a standalone clone.

The imported source identity and its verification are recorded in [additional/baseline](../additional/baseline/README.md). That import does not select a workload or substantiate a performance claim.
