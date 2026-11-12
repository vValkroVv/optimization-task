# HybridEP fused-scan fix implementation handoff

## Result

Implemented the campaign-only corrected entrypoint and adapter. The adapter
accepts only the pinned DeepEP wrapper (`63e425916b3c6b0c881ad402af35a2b00c4c84203fcde611a6715744b89210cb`)
and header (`b507f65cfa4580f16342076fc2d00afe81e43f64ddefcc4a3cf43681aef11b4b`),
stages one rank-local runtime copy, and changes only the fused-scan array bound
from `NUM_OF_EXPERTS_PER_RANK - 1` to `NUM_OF_EXPERTS_PER_RANK`. It loads the
staged wrapper under a unique `deep_ep.*` name and replaces only GIFT's
`torchtitan.distributed.deepep.hybridep.HybridEPBuffer` factory before buffer
construction. The original installed package and header remain untouched.

The entrypoint forwards the original argv object to native
`torchtitan.train.main`. The task runtime directory is required through
`GIFT_HYBRID_EP_SCAN_RUNTIME_DIR`; rank staging is selected by `RANK`. A used
rank directory is rejected. Each rank receives an empty cache root through the
existing `HYBRID_EP_CACHE_DIR`; the generated manifest records source/destination
hashes and modes, backend tree hashes, patch hashes, entry path, and native cache
path.

## Changed files

- `gift/torchtitan/experiments/cuda_graph_campaign/backend_scan_fix.py`
- `gift/torchtitan/experiments/cuda_graph_campaign/train_scan_fix.py`
- `gift/tests/unit_tests/test_campaign_backend_scan_fix.py`
- `gift/torchtitan/experiments/cuda_graph_campaign/backend_scan_fix.md`

## Verification

All checks ran CPU-only using `/opt/torchtitan-venv/bin/python` and `/opt/torchtitan-venv/bin/ruff`.

- Focused command: `PYTHONDONTWRITEBYTECODE=1 TMPDIR=<task>/runs/cuda-graphs-hybridep-fp8-h100/SCAN-FIX-IMPLEMENTATION/001/tmp /opt/torchtitan-venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=<task>/runs/cuda-graphs-hybridep-fp8-h100/SCAN-FIX-IMPLEMENTATION/001/tmp/pytest-final gift/tests/unit_tests/test_campaign_backend_scan_fix.py` — **11 passed**.
- `ruff check --no-cache` on the two Python source files and focused test — passed.
- `ruff format --check` on the same three Python files — passed.
- `ruff check --select F821,F811,F841 --no-cache` on the same three Python files — passed.
- Trailing-whitespace, final-newline and local Markdown-link check — passed.
- `git diff --check` — passed.
- A CPU-only import/call smoke also exercised the actual GIFT `get_buffer()` consumer with a fake native constructor and confirmed CUDA was not initialized.

The first `ruff format --check` reported two files needed formatting. They were
formatted, then the focused tests and final lint/format checks passed. No GPU,
NVCC, external package write, or training run was used.

## Cost and remaining gates

Expected host cost is one source-tree hash/copy, relocated import and fresh
native JIT compile per rank/process. The training update path adds no Python
branch, copy, synchronization, communication, or dtype change. The corrected
specialization adds one local integer to the fused scan; register allocation,
spills/local-memory use and device timing still require GPU inspection. This
source-level out-of-bounds defect is verified, but its causal connection to the
rank-2 illegal instruction remains unproven.

The 8-rank GPU smoke, strict deterministic A/A gate, numerical parity and
graph-off/graph-on performance measurements remain open for the execution
owner. Passing CPU staging/factory tests does not establish CUDA kernel,
distributed, numerical or performance acceptance.
