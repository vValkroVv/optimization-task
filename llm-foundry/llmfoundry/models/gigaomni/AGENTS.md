# GigaOmni production rules

These rules apply to code under `llmfoundry/models/gigaomni`.
Use [README.md](README.md) for the model lifecycle and [modalities/README.md](modalities/README.md) for modality, component, and `DataFlow` architecture.

## Production standard

- This is production model code for large-scale multimodal pretraining. Treat numerical behavior, checkpoint compatibility, distributed correctness, GPU memory, and throughput as first-class contracts.
- Fix the owning abstraction or contract. Do not add one-off conditionals, silent fallbacks, duplicated preprocessing, or downstream patches that hide an upstream mismatch.
- When porting an upstream model, preserve its reference behavior while retaining the optimized kernels already used by GigaOmni, `DataFlow`, GigaFSDP, sequence parallelism, meta initialization, and checkpoint contracts.
- Do not trade a measurable hot-path regression for local convenience. Measure representative shapes when a change can affect compute, memory, communication, or synchronization.

## Workflow

- Read the relevant `DataFlow` preset, component implementation, callers, neighboring implementation, and tests before changing behavior or interfaces.
- Trace a failure to the owning preset, component, or boundary before editing. Repair the source contract instead of compensating in unrelated downstream code.
- Make the smallest clear change at that owner and keep the touch set scoped to the request.
- Preserve unrelated working-tree changes.
- Store agent-created plans, reports, diagnostic scripts, temporary test harnesses, logs, and other task-local artifacts under `.agents/`, preferably `.agents/tasks/<task>/`; keep the whole directory out of Git.
- Update this file only when explicitly requested or when the task explicitly establishes a durable GigaOmni-wide convention.

## Code style

- Match import grouping and ordering in the nearest analogous GigaOmni file.
- Keep simple calls, assertions, conditions, comprehensions, and short signatures on one line when readable.
- Use multiline formatting for nested structures, many arguments, complex conditions, or expressions that are difficult to scan.
- Prefer clear names and the smallest code shape. Do not introduce pass-through helpers, aliases, wrappers, config objects, or classes unless they remove real duplication or isolate non-trivial logic.
- Do not narrate obvious code with comments or docstrings. Document non-obvious algorithms, tensor layouts, runtime constraints, and architectural decisions.
- Use `assert` only for internal contracts and invariants, with a concise diagnostic message.
- Preserve upstream module and parameter names when checkpoint compatibility depends on them.

## PyTorch and efficiency

- Keep training hot paths tensorized and device-resident. Batch or concatenate compatible routes and avoid per-sample or per-token module calls.
- Preserve existing fused kernels, FlashAttention, sequence parallelism, GigaFSDP boundaries, activation checkpointing, and mixed-precision behavior. Do not silently replace them with eager or Python fallbacks.
- Derive the device and dtype of new tensors from the relevant input or parameter. Do not introduce hard-coded CUDA devices or unconditional FP32 casts unless the contract requires them and tests cover the boundary.
- Treat `.item()`, `.tolist()`, `.cpu()`, tensor-dependent Python control flow, and host logging as synchronization points. Keep justified small-metadata transfers outside per-layer and other repeated hot paths.
- Do not add `.clone()`, `.contiguous()`, `.repeat()`, large temporary tensors, or extra collectives without a demonstrated layout, autograd, kernel, or communication requirement.
- Prefer functional tensor updates. Use in-place mutation only when ownership is unambiguous, autograd and checkpointing are safe, and the benefit is real; never use it to conceal a contract mismatch.

## DataFlow contract

- Treat `DataFlow` as ordered routing: `modality_inputs -> encoder_outputs -> adapter_outputs -> modality embeddings -> language model`.
- A component receives `List[Dict[str, Tensor]]` and returns `Tuple[Tensor, ...]`; each list item is one ordered route, not one batch sample, and each output corresponds positionally to that route.
- `_concat_inputs` validates route dictionaries, concatenates compatible tensors for one shared forward, and records enough boundary metadata to restore every route exactly.
- `_split_outputs` restores one tensor per original route and accounts explicitly for token-count changes such as spatial merging. `unwrap_tuple_output` only maps those tensors to the keys declared by `DataFlow`.
- Keep route order and produced keys unchanged through concatenation, compute, splitting, and unwrapping.
- A shared component runs once on concatenated compatible inputs. Final modality embeddings must have LLM hidden width and exactly match the corresponding placeholder-token count.

## GigaFSDP, initialization, and checkpoints

- A GigaFSDP wrap target must own every parameter used by its computation and access those parameters inside its `forward`, so hooks materialize and shard them at the correct boundary.
- Move upstream parameter-dependent logic into an appropriate wrapped module instead of accessing the parameter outside its owner. Preserve state-dict names such as `pos_embed.weight`.
- A custom wrap target with direct trainable parameters must implement `reset_parameters` for meta materialization.
- For a `PreTrainedModel`, initialize an unrecognized custom module through the same `reset_parameters` path in `_init_weights` and delegate other modules to `super()._init_weights(module)`.
- Recompute small deterministic derived tensors from scalar configuration on the runtime input device when materializing a buffer reliably would be harder and runtime cost is negligible.
- Register a buffer only when regular initialization, meta materialization, reset, checkpoint save, and checkpoint load paths are explicit and tested.

## Verification

- Before editing, identify the logical verification phases for the task. Run relevant checks after each completed phase—not after every micro-edit and not only once at the end.
- Run `ruff check` on changed Python files and the smallest relevant checks for the completed phase.
- Agents may write and run task-specific checks under `.agents/`, but must not commit them. Only new integration tests that follow the existing strict, pinned format may be committed.
- Test the owning boundary and failure paths; for shared contracts, cover multiple routes or modalities.
- For tensor or model changes, test representative shapes, dtypes, devices, forward and backward behavior, and exact output structure.
- For upstream ports, verify pinned-reference output and gradient parity with explicit tolerances using task-local checks.
- For initialization or parameter changes, test regular and meta initialization, materialization, state-dict keys, save/load behavior, and affected GigaFSDP wrap targets.
- For distributed-path changes, run the affected multi-rank sequence-parallel or GigaFSDP check and inspect every relevant rank.
- For hot-path changes, compare representative throughput and peak memory or provide profiler evidence; do not infer performance from code shape alone.
- Report commands and results precisely. A suite pass is not a successful cluster run until the job is terminal and retries and affected rank logs are clean.

## Execution and jobs

- Run checks directly from the current `llm-foundry` checkout.
- For a cluster train, conversion, or infinite job, copy `sber_scripts/submit_with_job_submitter.py` to a timestamped task-specific file under `sber_scripts/`, edit only the copy, and validate the effective job type, config, cluster login, worker and process counts, environment, and experiment name; never embed credentials.
- The launcher calls `job.submit()` at module scope. Do not import or execute it for validation; after submission is authorized, run `python sber_scripts/<timestamp>__<task>.py`.
- Inside an infinite allocation, inspect and configure `YAML_CFG`, `NNODES`, and `NGPUS_PER_NODE` in `sber_scripts/infinite_jobs/run.sh`, then launch it from the repository root with `bash sber_scripts/infinite_jobs/run.sh`.
- After an interrupted or failed infinite-job run, clean stale training processes and shared memory with `bash sber_scripts/infinite_jobs/clean.sh` before retrying.
- Record the submitted job ID and verify terminal status, retries, and every affected rank before reporting success.
