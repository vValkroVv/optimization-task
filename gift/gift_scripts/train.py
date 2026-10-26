"""Stable training entrypoint for the job submitter (``TrainJobGIFT``).

Runs from the checkout root so relative paths in configs resolve the same
way under the Cloud and Slurm launchers::

    train.py configs/experiments/foo.yaml [--section.key value ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    from torchtitan.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
