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
        exp_name="l16-st2-gbs-lr",
        baseline_config=BASELINE_REGISTRY / "giga/l16-11_3b-a-1_1b-4_0-papaya.yaml",
        baseline_patch=Path(__file__).with_name("baseline_patch.yaml"),
        setups=[
            Setup(
                run_name=f"gbs{gbs}-lr{label}",
                config_overrides={
                    "seed": 4096,
                    "global_train_batch_size": gbs,
                    "device_train_microbatch_size": mbs,
                    "model.config_overrides.expert_capacity_tokens": 16000 * mbs // 12,
                    "max_duration": f"{68700 * 768 // gbs}ba",
                    "scheduler.t_warmup": f"{1080 * 768 // gbs}ba",
                    "optimizer.lr": lr,
                    "optimizer.weight_decay": wd,
                    "save_interval": f"{687 * 768 // gbs}ba",
                    "stable_save_interval": f"{3435 * 768 // gbs}ba",
                    "console_log_interval": f"{10 * 768 // gbs}ba",
                    "callbacks.ema_weights.update_period": 5 * 768 // gbs,
                    "callbacks.ema_weights.start_batch": 6870 * 768 // gbs,
                },
            )
            # mbs 24 needs gbs >= 32 * 24, so smaller batches take a smaller mbs
            # 24mbs can cause OOM  during run resume, so I set 12 mbs
            # tbh in gift some things were done more optimal and we can use 24 mbs in test gift setups
            for gbs, mbs in ((192, 6), (384, 12), (768, 12))
            for label, lr, wd in (
                ("2p5e-4", 2.5e-4, 2.5e-5),
                ("3p5e-4", 3.5e-4, 3.5e-5),
                ("5e-4", 5e-4, 5e-5),
            )
        ],
        n_workers=4,
        n_gpus=8,
        max_retry=None,
        queue_name="pretrain" if PRIORITY_CLASS == "high" else None,
        priority_class=PRIORITY_CLASS,
        region="o5",
        submission_type="slurm",
    )


if __name__ == "__main__":
    main(build_experiment)
