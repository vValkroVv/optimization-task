import contextlib
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
MODEL_NAME = "l8-1_5b-a-0_3b"


if __name__ == "__main__":
    try:
        config_path, = REPO_ROOT.glob(f"*_configs/warmup/{MODEL_NAME}-deep-gemm.yaml")
        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                from contrib.job_submitter.job_submitter.jobs.llmf.warmup import WarmupJobLLMF

                jobs = WarmupJobLLMF.build_warmup_jobs(
                    user_login="profitrating",
                    exp_name=f"{MODEL_NAME}-deep-gemm-warmup",
                    warmup_type="deep_gemm",
                    warmup_config_path=config_path,
                    region="o5",
                    submission_type="slurm",
                    n_gpus=1,
                    n_workers=1,
                    priority_class="medium",
                    extra_env={"DG_CACHE_FORCE_SKIP": "1"},
                )
            for index, job in enumerate(jobs):
                if index:
                    time.sleep(1)
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    job.submit()
                print(f"Submitted warmup job {index + 1}/{len(jobs)}")
    except Exception:
        raise SystemExit("Warmup failed; reconcile existing jobs before retrying") from None
