import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB_SUBMITTER = ROOT / "contrib" / "job_submitter"
sys.path.insert(0, str(JOB_SUBMITTER))
sys.path.insert(0, str(JOB_SUBMITTER / "contrib" / "sqlite_s3_query"))

from job_submitter.jobs.image_registry import load_gift_base_images  # noqa: E402
from job_submitter.jobs.infinite import InfiniteJob  # noqa: E402

# --- edit these -------------------------------------------------------------
USER_LOGIN = "default"
EXP_NAME = "TorchTitan"
N_WORKERS = 2  # number of nodes
QUEUE_NAME = "pretrain-ml"
PRIORITY_CLASS = "medium"
# ---------------------------------------------------------------------------


class InfiniteJobTorchTitan(InfiniteJob):
    """Infinite job on the torchtitan image pinned in gift_docker/images.json."""

    BASE_IMAGES = load_gift_base_images(ROOT)


if __name__ == "__main__":
    InfiniteJobTorchTitan(
        user_login=USER_LOGIN,
        exp_name=EXP_NAME,
        n_workers=N_WORKERS,
        queue_name=QUEUE_NAME,
        priority_class=PRIORITY_CLASS,
    ).submit()
