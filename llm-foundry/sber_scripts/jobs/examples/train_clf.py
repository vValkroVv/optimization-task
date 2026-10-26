"""Submit gigaclf (LLM-based classifier) training via job-submitter.

Run from a submission env (client_lib + job_submitter deps) at the llm-foundry
root. ``TrainJobLLMF`` resolves region / base image / allocation and validates
the train config before submit. It derives the llm-foundry root from the
in-repo submodule (``contrib/job_submitter``), so run against the submodule copy.
"""

import sys
from pathlib import Path

REPO = Path(__file__).absolute().parents[3]  # -> llm-foundry root
sys.path.insert(0, str(REPO))

from contrib.job_submitter.job_submitter.jobs.llmf.train import TrainJobLLMF

CONFIG_PATH = REPO / "sber_configs" / "gigaclf" / "gigaclf_binary_config.yaml"


if __name__ == "__main__":
    assert CONFIG_PATH.exists(), f"config not found: {CONFIG_PATH}"

    TrainJobLLMF(
        user_login="<your-login>",
        exp_name="clf_exp",
        path_to_config=str(CONFIG_PATH),
        region="SR008",
        n_workers=1,
        n_gpus=8,
        # shared queue allows only priority_class="low"; for medium/high use a
        # dedicated queue (e.g. queue_name="pretrain-ml", priority_class="medium").
        queue_name="shared",
        priority_class="low",
        enable_auto_track=False,
        dry_run=True,  # validate config only; set False for a real submit
        max_retry=3,
    ).submit()
