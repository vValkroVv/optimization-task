"""Submit DeepGEMM JIT warmup jobs via job_submitter."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB_SUBMITTER = ROOT / "contrib" / "job_submitter"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(JOB_SUBMITTER))
sys.path.insert(0, str(JOB_SUBMITTER / "contrib" / "sqlite_s3_query"))

from job_submitter.jobs.gift.warmup import WarmupJobTorchTitan  # noqa: E402

from torchtitan.ops.warmup.plan import build_warmup_plan  # noqa: E402

USER_LOGIN = ""  # your login / telegram nickname
EXP_NAME = ""  # short label used in job description

# Path to the warmup config YAML, e.g. gift_configs/warmup/ultra_warmup_config.yaml.
# A relative path is resolved against the current directory on the submit host.
WARMUP_CONFIG_PATH = ""

# BASE_IMAGE = "" # use, if your image can invalid cache
QUEUE_NAME = "shared"


if __name__ == "__main__":
    assert USER_LOGIN, "USER_LOGIN must be set"
    assert WARMUP_CONFIG_PATH, "WARMUP_CONFIG_PATH must be set"

    # ROOT feeds both the command and the job's PYTHONPATH, so a pass cannot
    # end up running one checkout against another.
    plan = build_warmup_plan(WARMUP_CONFIG_PATH, gift_root=ROOT)
    plan.prepare_cache_dir()

    print(f"Submitting {len(plan.passes)} warmup job(s) -> {plan.cache_dir}")
    for warmup_pass in plan.passes:
        WarmupJobTorchTitan(
            user_login=USER_LOGIN,
            exp_name=EXP_NAME,
            job_label=warmup_pass.label,
            script=warmup_pass.command,
            gift_root_path=ROOT,
            queue_name=QUEUE_NAME,
        ).submit()
