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
        exp_name="l8-st1-mbs",
        baseline_config=BASELINE_REGISTRY / "giga/l8-1_5b-a-0_3b-4_0-papaya.yaml",
        baseline_patch=Path(__file__).with_name("baseline_patch.yaml"),
        setups=[
            Setup(
                run_name=f"mbs{mbs}-ec{capacity}",
                config_overrides={
                    "device_train_microbatch_size": mbs,
                    "model.config_overrides.expert_capacity_tokens": capacity,
                },
            )
            for mbs, capacity in ((3, 4000), (6, 8000), (12, 16000), (24, 32000))
        ],
        n_workers=1,
        n_gpus=8,
        queue_name="pretrain" if PRIORITY_CLASS == "high" else None,
        priority_class=PRIORITY_CLASS,
        region="o5",
        submission_type="slurm",
        extra_env={
            "PROPAGATE_DG_CACHE": "1",
            "PROPAGATE_DG_CACHE_DIR": "/mnt/nfs-extra/09-extra/pretrain/kernel-cache/l8-1_5b-a-0_3b-sm90-sms116",
        },
    )


if __name__ == "__main__":
    main(build_experiment)
