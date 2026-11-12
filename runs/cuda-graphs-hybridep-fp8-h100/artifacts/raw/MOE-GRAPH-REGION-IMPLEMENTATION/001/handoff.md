# Per-block CUDA Graph exclusion handoff

## Result

Implemented the opt-in `CompileConfig.cudagraph_exclude_blocks` list. Empty
preserves the existing per-block compile call. Nonempty lists require enabled
model compilation with the Inductor backend; duplicate names, invalid modes,
and unmatched block FQNs fail before compilation. Matched modules retain
`backend` and `fullgraph=True` and receive
`options={"triton.cudagraphs": False}`. Other blocks keep the canonical call.
The separate loss compile path is unchanged. Exclusions and their count/options
are logged once during model construction.

Files changed:

- `gift/torchtitan/config/configs.py`
- `gift/torchtitan/distributed/compile.py`
- `gift/tests/unit_tests/test_compile_cudagraph_exclusions.py`
- `gift/docs/config.md`
- 58 existing `gift/tests/unit_tests/golden_configs/experiments/*.yaml`
  snapshots, each with only the new empty default field
- `gift/tests/unit_tests/golden_configs/experiments/cuda_graph_campaign_continuation.yaml`,
  generated for the task-owned campaign YAML already in the workspace

The preset YAML was not changed. The config field stays a semantic compile
option (`compare=True`); no fingerprint behavior was hidden. The target is the
native trainer, whose checkpoints store model/optimizer/trainer state, while
resolved run config records the new field. The existing graph-trainer artifact
fingerprint explicitly hashes `mode`, `backend`, and `passes`; it is not used by
this target. The resolved-config round trip and affected config goldens cover
the serialized default.

## Integration and cost

The targeted meta-built GiftMoE 3B experiment yields `layers.0` through
`layers.17` through `walk_segments`; `peel` confirms layer 0 is dense and layers
1–17 contain the actual `HybridEPTokenDispatcher` at
`moe.experts.token_dispatcher`. Selective AC is applied before `apply_compile`,
and the test confirms the checkpoint wrappers themselves receive compile calls.
The fixture patches only the dispatcher constructor and `torch.compile`; it
does not initialize CUDA or run kernels.

The opt-in path adds one construction-time segment list/set validation and one
log entry. It adds no step-time work, tensors, arithmetic, synchronization,
copies, or collectives. Excluded blocks lose Inductor CUDA Graph replay benefit
and may incur additional host launches during training; magnitude is unmeasured.
Dense and loss compilation remain eligible under the inherited global policy.

## Verification

- Focused tests: 13 passed; guarded `is_available=False`, `device_count=0`, and
  failing `init`/`_lazy_init`; see `pytest-focused-003.log`.
- Regression sensitivity: with the exclusion option temporarily removed, the
  17-block behavior test failed on the expected missing `options`; source was
  restored before final checks; see `pytest-sensitivity-001.log`.
- Same-interpreter guard restoration identity check passed;
  `cuda-guard-restoration-001.log`.
- Established golden update path: 110 experiment-parity/model-architecture
  tests passed; `pytest-golden-repin-001.log`. Structural audit found exactly
  the empty default field in all 58 tracked experiment snapshots, no preset
  snapshot changes, and the new continuation snapshot contains no personal
  paths; `golden-diff-audit-001.log`.
- Required Ruff selection passed; `ruff-focused-002.log`.
- `git diff --check` passed; `diff-check-001.log`.
- Local document-link check passed for 28 inline links;
  `docs-links-001.log`.

No CUDA query, initialization, capture, training run, or performance test was
performed. A005 remains failure evidence; native A006 capture and performance
acceptance remain open for the coordinator's campaign gate.
