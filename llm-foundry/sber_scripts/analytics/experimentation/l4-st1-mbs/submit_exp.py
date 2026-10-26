import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sber_scripts.analytics.experimentation import Experiment, Setup
from sber_scripts.analytics.experimentation.run import main
from sber_scripts.analytics.experimentation.submit import BASELINE_REGISTRY

PRIORITY_CLASS = "medium"
N_GPUS = 2


def build_experiment():
    return Experiment(
        user_login="vapoletaev",
        exp_name="l4-cloud-st1-mbs",
        baseline_config=BASELINE_REGISTRY / "giga/l4-0_25b-a-0_13b-4_0-cloud.yaml",
        baseline_patch=Path(__file__).with_name("baseline_patch.yaml"),
        setups=[
            Setup(
                run_name=f"mbs{mbs}-ec{capacity}",
                config_overrides={
                    "global_train_batch_size": 2 * mbs,
                    "device_train_microbatch_size": mbs,
                    "model.config_overrides.expert_capacity_tokens": capacity,
                },
            )
            for mbs, capacity in (
                (32, 48000),
                (48, 72000),
                (56, 84000),
                (64, 96000),
            )
        ],
        n_workers=1,
        n_gpus=N_GPUS,
        processes_per_worker=N_GPUS,
        queue_name="pretrain-ml",
        priority_class=PRIORITY_CLASS,
        extra_env={"DG_CACHE_FORCE_SKIP": "1"},
    )


if __name__ == "__main__":
    main(build_experiment)
