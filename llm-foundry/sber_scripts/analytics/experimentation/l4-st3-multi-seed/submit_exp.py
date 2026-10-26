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
        exp_name="l4-cloud-st3-multi-seed",
        baseline_config=BASELINE_REGISTRY / "giga/l4-0_25b-a-0_13b-4_0-cloud.yaml",
        baseline_patch=Path(__file__).with_name("baseline_patch.yaml"),
        setups=[
            Setup(run_name=f"seed{seed}", config_overrides={"seed": seed})
            for seed in (1024, 2048, 4096, 8192, 16384)
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
