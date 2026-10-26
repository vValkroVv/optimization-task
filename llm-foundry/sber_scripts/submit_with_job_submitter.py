# ruff: noqa: E402
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR / "contrib" / "job_submitter"))

from job_submitter.jobs.llmf.train import TrainJobLLMF
from job_submitter.jobs.infinite import InfiniteJob
from job_submitter.jobs.gigachat_utils.test_convertation import (
    TestConvertationJobGigachatUtils,
)


N_WORKERS = 0  # number of nodes
USER = "your_mm_username"  # your mm login
EXP_NAME = "new_sota"
JOB_TYPE = "your_job_type"  # "train" | "inf" | "convert"
CONFIG = ""  # path to merged training config or convertation config
EXTRA_ENV = {
    "AWS_ACCESS_KEY_ID": "...",
    "AWS_SECRET_ACCESS_KEY": "...",
    "S3_ADDRESSING_STYLE": "virtual",
}  # s3 credentials for training from s3

if JOB_TYPE == "train":
    job = TrainJobLLMF(
        user_login=USER,
        exp_name=EXP_NAME,
        path_to_config=CONFIG,
        script_type="pytorch2",
        use_clearml=False,
        script_args="",
        processes_per_worker=8,
        n_workers=N_WORKERS,
        extra_env=EXTRA_ENV,
    )
elif JOB_TYPE == "inf":
    job = InfiniteJob(
        user_login=USER,
        exp_name=f"inf job {EXP_NAME}",
        processes_per_worker=8,
        n_workers=1,
    )
elif JOB_TYPE == "convert":
    convertation_script_path = (
        f"{ROOT_DIR}/contrib/gigachat-utils/convertation/map-fsdp-to-fsdp.py"
    )
    script = f"python {convertation_script_path} --input-config-path {CONFIG}"
    job = TestConvertationJobGigachatUtils(
        user_login=USER,
        exp_name=f"convertation job {EXP_NAME}",
        script=script,
        processes_per_worker=1,
        n_workers=1,
        n_gpus=1,
    )
result = job.submit()
print(result)
