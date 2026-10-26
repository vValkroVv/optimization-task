import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_SUBMITTER_ROOT = REPO_ROOT / "contrib" / "job_submitter"
sys.path.insert(0, str(JOB_SUBMITTER_ROOT))

from job_submitter.jobs.gift.train import TrainJobGIFT  # noqa: E402

if __name__ == "__main__":
    TrainJobGIFT(
        user_login="<user_login>",
        exp_name="<exp_name>",
        # An experiment YAML (absolute or relative to the checkout root) plus dot-path
        # overrides; validate first with `python scripts/config_check.py <yaml>`.
        config_file="configs/experiments/gift_moe_debug.yaml",
        script_args="",
        n_workers=1,
        queue_name="pretrain-ml",
        priority_class="medium",
        gift_root_path=REPO_ROOT,
    ).submit()
