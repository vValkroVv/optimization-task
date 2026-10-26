import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent.parent))

from contrib.job_submitter.job_submitter.jobs.any_script import AnyScriptJob
from contrib.job_submitter.job_submitter.jobs.image_registry import LLMF_BASE_IMAGES

LLM_F_ROOT = Path(__file__).absolute().parent.parent.parent.parent

USER_LOGIN = ""

if __name__ == "__main__":
    wrapper = LLM_F_ROOT / "papaya_scripts" / "jobs" / "nsys_rank_wrapper.py"
    # Add in the callback section in the config:
    #nsys_capture:
    #  start_batch: 3
    #  stop_batch: 5
    config = "<path to config>"

    script = (
        f"{wrapper} "
        f"{config} "
        f"use_clearml=false "
        f"use_profiler=false"
    )

    contrib_path = LLM_F_ROOT / "contrib"
    contrib_dirs = [str(p) for p in contrib_path.iterdir() if p.is_dir()] if contrib_path.exists() else []

    extra_env = {
        "PYTHONPATH": ":".join([str(LLM_F_ROOT)] + contrib_dirs),
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "8",
        "PROPAGATE_DG_CACHE": 1,
        "PROPAGATE_DG_CACHE_DIR": "/mnt/nfs/07-share/ivlog7/KERNEL_CACHE_SM116",
        "NUM_DEEP_GEMM_SMS": 116,
    }

    AnyScriptJob(
        user_login=USER_LOGIN,
        exp_name=f"{USER_LOGIN} nsys profile",
        script=script,
        script_type="pytorch2",
        n_gpus=8,
        n_workers=64,
        region="o1",
        enable_auto_track=True,  # 🚀 Want this run to show up in Mattermost automatically? ✨
        # ✨ Set Auto-tracking to True and you will get notifications in mattermost! 💪💪💪
        # if you submit from slurm and doesn't get notifications check yourself in mapping
        # Mapping of username nicknames is here in gitlab
        # papayallms/papayachat_tracker_bot/-/blob/main/configs/username_mapping.json
        # don't forget to change papaya to ... not papaya. 🥴🍑
        # and don't forget to set your correct mattermost account or papaya slurm name
        priority_class="high",
        base_image=LLMF_BASE_IMAGES["o5"],
        job_logdir=f"/mnt/nfs/07-share/{USER_LOGIN}/jobs_logs/nsys_profile",
        slurm_logdir=f"/mnt/nfs/07-share/{USER_LOGIN}/slurm_logs/nsys_profile",
        max_retry=None,
        extra_env=extra_env,
        queue_name='pretrain-ml'
    ).submit()
