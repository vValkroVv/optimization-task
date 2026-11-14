# Combined-four integration checks

14 November 2026. Scope: transient integration of the existing routing-metadata
and stable-CE activation patches in task-local `gift/`, then focused composition
checks and exact baseline restoration. No native training or performance run
was launched.

## Result

Both patches applied together and changed only
`gift/torchtitan/distributed/deepep/hybridep.py` and
`gift/torchtitan/components/loss.py`. Their source diff passed `git diff
--check`. The focused CUDA checks passed: three selected cases, four unrelated
MTP parameter cases deselected, zero skips, and one environment deprecation
warning for `pynvml`.

The patches were reverse-applied after the checks. The two restored core-file
SHA256 values match the frozen baseline exactly. The exact applied diff and all
source/helper/test identities are in [source-identities.json](source-identities.json);
the byte-for-byte candidate diff is [candidate-core.patch](candidate-core.patch).

## Focused checks

The existing WM profile was readable and its hash is recorded in the identity
manifest. The selected runtime was
`/opt/torchtitan-venv/bin/python3.12` (Python 3.12.3, Torch
2.15.0.dev20260822+cu130, CUDA 13.0, Triton 3.8.0). Eight H100 GPUs were visible
and idle before execution. The tests ran on the current local allocation; no
allocation or remote connection was requested.

Command, from the task-local `gift/` directory:

```bash
bash -c 'set -Eeuo pipefail; source /etc/profile.d/wm.sh; ulimit -c 0; export PYTHONPATH=/workspace-SR008.nfs2/vkropotin/optimization-task/gift; export PYTHONDONTWRITEBYTECODE=1; /opt/torchtitan-venv/bin/python3.12 -m pytest -q tests/unit_tests/test_fusion_routing_metadata.py::test_triton_metadata_runs_inside_hybridep_dispatch_custom_op tests/unit_tests/test_fusion_losses.py::test_native_mtp_two_chunks_loss_hidden_weight_and_identity -k "test_triton_metadata_runs_inside_hybridep_dispatch_custom_op or native_float"'
```

Result: `3 passed, 4 deselected, 1 warning in 8.33s`. The routing case ran
through the active HybridEP dispatch custom op. The two selected MTP cases used
the native-float denominator with eager and compiled loss paths. No CUDA or
inactive-patch skip occurred. Four other parameter combinations in the named
MTP test were intentionally deselected.

The diagnostic protocol sets `--capture-microbatches` only for update 21 so the
existing measurement receipt can confirm all 32 completed microbatch calls;
trace collection itself remains capped at three microbatches on ranks 0 and 1.
Timing-run counters, profiling and microbatch capture remain off.

Scoped lint command, run while the candidate source was active:

```bash
ruff check --select F821,F811,F841 --no-cache gift/torchtitan/distributed/deepep/hybridep.py gift/torchtitan/components/loss.py
```

Result: `All checks passed!` The joint patch applicability check passed before
application, and the reverse application restored the frozen baseline hashes.
Argument construction used `run._parse_args` → `run._command` →
`measure._parse_args`; neither `main()` nor a training process was invoked.
Exact argv and parsed helper arguments for the baseline, candidate and
diagnostic are in [protocol.json](protocol.json).

## Remaining gate

These checks establish the two new binding points compose under their existing
GPU tests. They do not establish activation through the full native training
path, update-boundary cache behavior, combined timing, memory, convergence, or
distributed production behavior. The authorized native baseline, candidate and
diagnostic gates remain open.
