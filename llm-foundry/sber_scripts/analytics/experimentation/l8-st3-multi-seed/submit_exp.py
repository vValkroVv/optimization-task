import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sber_scripts.analytics.experimentation import Experiment, Setup
from sber_scripts.analytics.experimentation.run import main
from sber_scripts.analytics.experimentation.submit import BASELINE_REGISTRY

PRIORITY_CLASS = "medium"


def build_experiment():
    return Experiment(
        user_login="profitrating",
        exp_name="l8-st3-multi-seed",
        baseline_config=BASELINE_REGISTRY / "giga/l8-1_5b-a-0_3b-4_0-papaya.yaml",
        baseline_patch=Path(__file__).with_name("baseline_patch.yaml"),
        setups=[
            Setup(run_name=f"seed{seed}", config_overrides={"seed": seed})
            for seed in (1024, 2048, 4096, 8192, 16384)
        ],
        n_workers=1,
        n_gpus=8,
        queue_name="pretrain" if PRIORITY_CLASS == "high" else None,
        priority_class=PRIORITY_CLASS,
        region="o5",
        submission_type="slurm",
    )


if __name__ == "__main__":
    main(build_experiment)
