"""Submit Triton autotune or DeepGEMM JIT warmup jobs via job_submitter.

Usage:
    1. Fill in the settings below.
    2. python sber_scripts/jobs/examples/warmup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent.parent))
from contrib.job_submitter.job_submitter.jobs.llmf.warmup import WarmupJobLLMF

#################################################################################
# ── USER SETTINGS ──────────────────────────────────────────────────────────────
#################################################################################

USER_LOGIN = ""  # your login / telegram nickname
EXP_NAME = ""  # short label used in job description

# Which warmup to run: "triton" or "deep_gemm"
WARMUP_TYPE = "deep_gemm"

# Absolute path to warmup config YAML (e.g. sber_configs/warmup/ultra_warmup_config.yaml)
WARMUP_CONFIG_PATH = ""

# ── Optional overrides (leave empty for auto-detection) ───────────────────────
QUEUE_NAME = "shared"

#################################################################################

if __name__ == "__main__":
    assert USER_LOGIN, "USER_LOGIN must be set"
    assert WARMUP_CONFIG_PATH, "WARMUP_CONFIG_PATH must be set"

    jobs = WarmupJobLLMF.build_warmup_jobs(
        user_login=USER_LOGIN,
        exp_name=EXP_NAME,
        warmup_type=WARMUP_TYPE,
        warmup_config_path=WARMUP_CONFIG_PATH,
        queue_name=QUEUE_NAME,
        extra_env={
            "DG_CACHE_FORCE_SKIP": "1",
        }
    )

    print(f"Submitting {len(jobs)} warmup job(s)  [type={WARMUP_TYPE}]")
    for job in jobs:
        job.submit()
