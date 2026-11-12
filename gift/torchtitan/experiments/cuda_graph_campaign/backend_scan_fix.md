# Campaign HybridEP fused-scan correction

`train_scan_fix` is an opt-in entrypoint for the CUDA Graph campaign. It stages
the installed DeepEP HybridEP wrapper and backend into a new rank-specific
directory, then changes only the fused-scan local-array bound from
`NUM_OF_EXPERTS_PER_RANK - 1` to `NUM_OF_EXPERTS_PER_RANK`. The installed
wrapper and package remain untouched. The wrapper is loaded from the staged
path under a unique `deep_ep.*` module name, so its existing native constructor
passes the staged directory as `base_path`.

The entrypoint validates the installed wrapper and header SHA-256 values before
copying. It writes a manifest with source and staged file hashes and modes,
backend tree hashes, the one-line correction, entry path, and the native JIT
cache paths. Each rank requires its own `RANK` value. An existing rank
directory is rejected; use a fresh attempt root for every new process launch.
The adapter sets `HYBRID_EP_CACHE_DIR` to an empty rank-local cache root because
the native cache key does not include header contents. The buffer must not have
been constructed before the adapter runs.

Example, with a new attempt directory supplied by the campaign runner:

```bash
export GIFT_HYBRID_EP_SCAN_RUNTIME_DIR="${CAMPAIGN_RAW_DIR}/scan-fix-attempt-002"
torchrun --nproc_per_node=8 -m torchtitan.experiments.cuda_graph_campaign.train_scan_fix \
  configs/experiments/cuda_graph_campaign_continuation.yaml
```

The entrypoint forwards the argument sequence unchanged to `torchtitan.train.main`;
the native trainer still owns configuration, model and loss construction,
distributed setup, training, checkpointing, and shutdown. The regular
`torchtitan.train` entrypoint remains the unmodified baseline route.
It initializes GIFT's console logger before installing the adapter so each
rank's activation marker is visible before native training begins.
For the corrected candidate, use this entrypoint for both graph-off and
graph-on runs, changing only the campaign's existing graph setting.

The installed fused scan allocates `NUM_OF_EXPERTS_PER_RANK - 1` integers but
indexes through `NUM_OF_EXPERTS_PER_RANK - 1`. The upstream correction changes
that allocation to `NUM_OF_EXPERTS_PER_RANK`. This establishes a source-level
out-of-bounds access for the fused path; it does not establish that this defect
caused the observed rank-2 illegal instruction. The correction adds one local
integer to the affected specialization, so generated code, registers, local
memory and runtime cost still require GPU inspection. CPU tests cover staging,
factory replacement, and the native base-path argument only; they do not
establish CUDA kernel correctness, numerical parity, or distributed acceptance.
