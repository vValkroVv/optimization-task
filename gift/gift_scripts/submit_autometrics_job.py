import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "contrib" / "job_submitter"))

from job_submitter.jobs.gift.autometrics import AutometricsJobGIFT  # noqa: E402

if __name__ == "__main__":
    AutometricsJobGIFT(
        user_login="<user_login>",
        exp_name="<exp_name>",
        checkpoint_dir="<path to complete HF checkpoint>",
        path_to_config="<path to PAM config>",
        n_workers=1,
        enable_auto_track=True,
        queue_name=None,
        priority_class=None,
        gift_root_path=REPO_ROOT,
    ).submit()
