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
        exp_name="l4-cloud-st2-gbs-lr",
        baseline_config=BASELINE_REGISTRY / "giga/l4-0_25b-a-0_13b-4_0-cloud.yaml",
        baseline_patch=Path(__file__).with_name("baseline_patch.yaml"),
        setups=[
            Setup(
                run_name=f"gbs{gbs}-lr{label}",
                config_overrides={
                    "seed": 4096,
                    "global_train_batch_size": gbs,
                    "max_duration": f"{56000 * 112 // gbs}ba",
                    "scheduler.t_warmup": f"{1000 * 112 // gbs}ba",
                    "optimizer.lr": lr,
                    "optimizer.weight_decay": wd,
                    "save_interval": f"{560 * 112 // gbs}ba",
                    "stable_save_interval": f"{5 * (560 * 112 // gbs)}ba",
                    "callbacks.autometrics.eval_interval": f"{5 * (560 * 112 // gbs)}ba",
                    "console_log_interval": f"{10 * 112 // gbs}ba",
                    "callbacks.ema_weights.update_period": 5 * 112 // gbs,
                    "callbacks.ema_weights.start_batch": 5600 * 112 // gbs,
                },
            )
            for gbs in (112, 224, 448)
            for label, lr, wd in (
                ("1e-3", 1e-3, 1e-4),
                ("1p4e-3", 1.4e-3, 1.4e-4),
                ("7e-4", 7e-4, 7e-5),
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
