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
        exp_name="l8-st2-gbs-lr",
        baseline_config=BASELINE_REGISTRY / "giga/l8-1_5b-a-0_3b-4_0-papaya.yaml",
        baseline_patch=Path(__file__).with_name("baseline_patch.yaml"),
        setups=[
            Setup(
                run_name=f"gbs{gbs}-lr{label}",
                config_overrides={
                    "seed": 4096,
                    "global_train_batch_size": gbs,
                    "max_duration": f"{40000 * 384 // gbs}ba",
                    "scheduler.t_warmup": f"{1000 * 384 // gbs}ba",
                    "optimizer.lr": lr,
                    "optimizer.weight_decay": wd,
                    "save_interval": f"{400 * 384 // gbs}ba",
                    "stable_save_interval": f"{2000 * 384 // gbs}ba",
                    "console_log_interval": f"{10 * 384 // gbs}ba",
                    "callbacks.ema_weights.update_period": 5 * 384 // gbs,
                    "callbacks.ema_weights.start_batch": 4000 * 384 // gbs,
                },
            )
            for gbs in (96, 192, 384)
            for label, lr, wd in (
                ("7e-4", 7e-4, 7e-5),
                ("1e-3", 1e-3, 1e-4),
                ("1p4e-3", 1.4e-3, 1.4e-4),
            )
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
