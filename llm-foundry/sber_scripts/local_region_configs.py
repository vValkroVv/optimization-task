import sys
from pathlib import Path
from typing import List


def update_sys_path(paths: List[str]) -> None:
    sys.path.extend([str(Path(path).absolute()) for path in paths])


update_sys_path(
    [str(Path(__file__).parent.parent / "contrib/gigachat-utils/job_templates/src")]
)

# ruff: noqa: E402
from job_templates import (
    SLURM_REGIONS,
    CLOUD_REGIONS,
    DEFAULT_REGION_ENVS,
    DEFAULT_SLURM_REGION_PARAMS,
    get_instance_type,
)

get_instance_type = get_instance_type  # Для реимпорта в другие скрипты


# default env variable (slurm & cloud)
_DEFAULT_HOME = "/home/user/"

DEFAULT_JOB_DISPATCHER_API_ENVS = {
    "JOB_DISPATCHER_BASE_URL": "https://jdmetrics.sberdevices.ru/api",
    "JOB_DISPATCHER_SLURM_BASE_URL": "https://dj.static.space/api",
    "JOB_DISPATCHER_TOKEN": "YlhIrKHpvuv0pBb6CK95kFv2AADp/79EUJ+njxqtSqk=",
}

DEFAULT_JOB_DISPATCHER_S3_ENVS = {
    "JOB_DISPATCHER_S3_DEST": "gigaeye-s3:gigaeye-pretrain-checkpoints/jd_artifacts",
    "JOB_DISPATCHER_RCLONE_PATH": "/workspace/job-dispatcher-utils/s3-interaction/rclone",
    "JOB_DISPATCHER_RCLONE_CONFIG": "/workspace/job-dispatcher-utils/s3-interaction/rclone.conf",
    "JOB_DISPATCHER_S3_PATH_MARKER": "gigachat_checkpoints/",
}

LOCAL_DEFAULT_REGION_ENVS = {
    SLURM_REGIONS.o3_64: {
        "HOME": _DEFAULT_HOME,
        "PYTORCH_KERNEL_CACHE_PATH": "/tmp/.cache",
        "TRITON_CACHE_DIR": "/tmp/.cache/triton/\\$LOCAL_RANK",
        "ENV_FILEPATH_WANDB": "/mnt/fl/.wandb.env",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/mnt/fl/01-home/infidelis/.cache/benches",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
    },
    SLURM_REGIONS.o3_128: {
        "HOME": _DEFAULT_HOME,
        "PYTORCH_KERNEL_CACHE_PATH": "/tmp/.cache",
        "TRITON_CACHE_DIR": "/tmp/.cache/triton/\\$LOCAL_RANK",
        "ENV_FILEPATH_WANDB": "/mnt/nfs/01-home/infidelis/.wandb.env",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/mnt/nfs/01-home/infidelis/.cache/benches",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
    },
    SLURM_REGIONS.o4: {
        "HOME": _DEFAULT_HOME,
        "PYTORCH_KERNEL_CACHE_PATH": "/tmp/.cache",
        "TRITON_CACHE_DIR": "/tmp/.cache/triton/\\$LOCAL_RANK",
        "ENV_FILEPATH_WANDB": "/mnt/nfs/01-home/loggerhead/.wandb.env",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/mnt/nfs/01-home/loggerhead/.cache/benches",
    },
    SLURM_REGIONS.o1: {
        "HOME": _DEFAULT_HOME,
        "PYTORCH_KERNEL_CACHE_PATH": "/tmp/.cache",
        "TRITON_CACHE_DIR": "/tmp/.cache/triton/\\$LOCAL_RANK",
        "ENV_FILEPATH_WANDB": "/mnt/nfs/01-home/infidelis/.wandb.env",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/mnt/nfs/01-home/infidelis/.cache/benches",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
    },
    SLURM_REGIONS.o5: {
        "HOME": _DEFAULT_HOME,
        "PYTORCH_KERNEL_CACHE_PATH": "/tmp/.cache",
        "TRITON_CACHE_DIR": "/tmp/.cache/triton/\\$LOCAL_RANK",
        "ENV_FILEPATH_WANDB": "/mnt/nfs/01-home/infidelis/.wandb.env",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/mnt/nfs/01-home/infidelis/.cache/benches",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
    },
    CLOUD_REGIONS.sr004: {
        # specify wandb by user
        "WANDB_BASE_URL": "https://wandb.sberdevices.ru",
        "WANDB_INSECURE_DISABLE_SSL": "true",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/home/jovyan/.cache/benches",
        "JOB_DISPATCHER_CALLBACK_SIGNAL_FILES": "/home/jovyan/.pretrain_utils/pam-autoconfigs",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
        **DEFAULT_JOB_DISPATCHER_S3_ENVS,
    },
    CLOUD_REGIONS.sr006: {
        # specify wandb by user
        "WANDB_BASE_URL": "http://wandb.su:8081",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/home/jovyan/.cache/benches",
        "JOB_DISPATCHER_CALLBACK_SIGNAL_FILES": "/home/jovyan/.pretrain_utils/pam-autoconfigs",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
        **DEFAULT_JOB_DISPATCHER_S3_ENVS,
    },
    CLOUD_REGIONS.sr008: {
        # specify wandb by user
        "WANDB_BASE_URL": "http://wandb.su:8081",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/workspace-SR008.fs2/.cache/benches",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
        **DEFAULT_JOB_DISPATCHER_S3_ENVS,
    },
}

for region in DEFAULT_REGION_ENVS:
    DEFAULT_REGION_ENVS[region].update(LOCAL_DEFAULT_REGION_ENVS[region])

# default images and log paths
LOCAL_DEFAULT_SLURM_REGION_PARAMS = {
    SLURM_REGIONS.o3_64: {
        "default_image_path": "/mnt/fl/sw/images/llm-foundry-train_2024-09-12_commit47444564_v010.sqsh"
    },
    SLURM_REGIONS.o3_128: {
        "default_image_path": "/mnt/nfs/03-image/llm-foundry-train_2024-09-12_commit47444564_v010.sqsh"
    },
    SLURM_REGIONS.o4: {
        "default_image_path": "/mnt/nfs/01-home/loggerhead/images/llm-foundry-train_2024-09-12_commit47444564_v010.sqsh"
    },
    SLURM_REGIONS.o1: {
        "default_image_path": "/mnt/nfs/01-home/infidelis/transfer2papaya/images/llm-foundry-train_2024_11_19_commit526b2c83-sm90.sqsh"
    },
    SLURM_REGIONS.o5: {
        "default_image_path": "/mnt/nfs/01-home/infidelis/transfer2papaya/images/llm-foundry-train_2024_11_19_commit526b2c83-sm90.sqsh"
    },
}

for region in DEFAULT_SLURM_REGION_PARAMS:
    DEFAULT_SLURM_REGION_PARAMS[region].update(
        LOCAL_DEFAULT_SLURM_REGION_PARAMS[region]
    )

# default image for all cloud regions
_DEFAULT_CLOUD_IMAGE = "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/llm-foundry:2026_03_17_commitba09621f"

DEFAULT_CLOUD_REGION_IMAGES = {
    CLOUD_REGIONS.sr004: _DEFAULT_CLOUD_IMAGE,
    CLOUD_REGIONS.sr006: _DEFAULT_CLOUD_IMAGE,
    CLOUD_REGIONS.sr008: _DEFAULT_CLOUD_IMAGE,
}
