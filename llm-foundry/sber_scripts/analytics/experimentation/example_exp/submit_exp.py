import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from sber_scripts.analytics.experimentation import Experiment, Setup  # noqa: E402

SETUPS = [
    Setup(
        run_name=...,
        config_overrides={
            "global_seed": 2048,
            "train_loader.dataset.shuffle_seed": 2048,
        },
    )
]

if __name__ == "__main__":
    experiment = Experiment(
        user_login=...,
        exp_name=...,
        baseline_config=...,  # Full config from sber_configs/baseline_registry/.
        baseline_patch=...,  # baseline_patch.yaml from this directory.
        setups=SETUPS,
        n_workers=1,
        enable_auto_track=True,
        max_retry=None,
        job_logdir=None,
        slurm_logdir=None,
        extra_env={},
        queue_name=None,
        priority_class=None,
    )
    experiment.submit()
