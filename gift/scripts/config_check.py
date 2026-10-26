"""Validate an experiment YAML config before submitting it to the cluster.

Thin wrapper over ``torchtitan.config.pre_submit`` (also the ``config-check``
console script); needs only Python and PyYAML, no torch.

::

    python scripts/config_check.py configs/experiments/llama3_debug.yaml \\
        --training.steps 2000 --optimizer.lr 5e-5 --print
"""

import sys
from pathlib import Path

# scripts/ is not a package: put the repo root on sys.path for a bare checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torchtitan.config.pre_submit import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
