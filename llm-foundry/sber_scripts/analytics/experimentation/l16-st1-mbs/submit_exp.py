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
        exp_name="l16-st1-mbs",
        baseline_config=BASELINE_REGISTRY / "giga/l16-11_3b-a-1_1b-4_0-papaya.yaml",
        baseline_patch=Path(__file__).with_name("baseline_patch.yaml"),
        setups=[
            Setup(
                run_name=f"mbs{mbs}-ec{capacity}" + ("-ac" if ac else ""),
                config_overrides={
                    "device_train_microbatch_size": mbs,
                    "model.config_overrides.expert_capacity_tokens": capacity,
                    "papayafsdp_config.activation_checkpointing": ac,
                },
            )
            for mbs, capacity, ac in (
                (3, 4000, False),
                (12, 16000, True),
                (24, 32000, True),
            )
        ],
        n_workers=4,
        n_gpus=8,
        queue_name="pretrain" if PRIORITY_CLASS == "high" else None,
        priority_class=PRIORITY_CLASS,
        region="o5",
        submission_type="slurm",
    )


if __name__ == "__main__":
    main(build_experiment)
