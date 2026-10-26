class SLURM_REGIONS:
    o1 = "o1"
    o5 = "o5"
    o2 = "o2"


SLURM_TEAMS = ("pretrain", "data", "alignment")
# Clusters where the team partition (`<team>`) plus team QoS classes
# (`<team>_high` / `<team>` / `<team>_low` / `normal`) replace `main`/`middle`.
SLURM_TEAM_QOS_REGIONS = frozenset({SLURM_REGIONS.o1, SLURM_REGIONS.o2})


class CLOUD_REGIONS:
    sr004 = "SR004"
    sr006 = "SR006"
    sr008 = "SR008"


_DEFAULT_HOME = "/home/user/"
DEFAULT_JOB_DISPATCHER_BASE_URL = "https://jdmetrics.sberdevices.ru/api"
DEFAULT_JOB_DISPATCHER_SLURM_BASE_URL = "https://dj.static.space/api"
DEFAULT_JOB_DISPATCHER_TIMEOUT_SECONDS = 15.0
DEFAULT_JOB_DISPATCHER_TOKEN = "YlhIrKHpvuv0pBb6CK95kFv2AADp/79EUJ+njxqtSqk="
JOB_DISPATCHER_BASE_URL_ENV = "JOB_DISPATCHER_BASE_URL"
JOB_DISPATCHER_SLURM_BASE_URL_ENV = "JOB_DISPATCHER_SLURM_BASE_URL"
JOB_DISPATCHER_TIMEOUT_SECONDS_ENV = "JOB_DISPATCHER_TIMEOUT_SECONDS"
JOB_DISPATCHER_TOKEN_ENV = "JOB_DISPATCHER_TOKEN"
DEFAULT_JOB_DISPATCHER_S3_DEST = "gigaeye-s3:gigaeye-pretrain-checkpoints/jd_artifacts"
DEFAULT_JOB_DISPATCHER_SLURM_S3_DEST = "papayaeye-papaya:jd-artifacts"
JOB_DISPATCHER_S3_DEST_ENV = "JOB_DISPATCHER_S3_DEST"

DEFAULT_JOB_DISPATCHER_API_ENVS = {
    JOB_DISPATCHER_BASE_URL_ENV: DEFAULT_JOB_DISPATCHER_BASE_URL,
    JOB_DISPATCHER_SLURM_BASE_URL_ENV: DEFAULT_JOB_DISPATCHER_SLURM_BASE_URL,
    JOB_DISPATCHER_TOKEN_ENV: DEFAULT_JOB_DISPATCHER_TOKEN,
}

DEFAULT_JOB_DISPATCHER_S3_ENVS = {
    JOB_DISPATCHER_S3_DEST_ENV: DEFAULT_JOB_DISPATCHER_S3_DEST,
    "JOB_DISPATCHER_RCLONE_PATH": "/workspace/job-dispatcher-utils/s3-interaction/rclone",
    "JOB_DISPATCHER_RCLONE_CONFIG": "/workspace/job-dispatcher-utils/s3-interaction/rclone.conf",
}

TRACKING_AUTO_TRACK_DB_LOCAL_DIR_ENV = "TRACKING_AUTO_TRACK_DB_LOCAL_DIR"
TRACKING_AUTO_TRACK_DB_LOCAL_DIR_PREFIX = "tracker_db_autotrack_"
TRACKING_S3_ENV_FILE_ENV = "TRACKING_S3_ENV_FILE"
TRACKING_S3_SECRET_ENV_FILE_KEY = "ENV_FILEPATH_S3_DB"
TRACKING_REQUIRED_SECRET_PATH_MARKER = "super_secret"
TRACKING_AUTO_TRACK_ERRORS_COMMAND_TYPE = "auto_track_errors"
TRACKING_LAST_CHECKED_RETRY_DEFAULT = 0
# S3 connection details are cluster-specific and provided at runtime via the
# per-cluster secret env file (see ``ENV_FILEPATH_S3_DB``). The repository ships
# no concrete endpoint/region/bucket so it stays portable when transferred
# between clusters; ``us-east-1`` is only a neutral placeholder for
# S3-compatible providers that still require a region name.
TRACKING_DEFAULT_S3_ENDPOINT_URL = ""
TRACKING_DEFAULT_S3_REGION = "us-east-1"
TRACKING_DEFAULT_S3_BUCKET = ""
TRACKING_DEFAULT_S3_KEY_PREFIX = "tracker/"
# Environment keys that let a cluster's secret env file opt into the on-prem
# S3 gateway behaviour (self-signed TLS + path-style addressing).
TRACKING_S3_VERIFY_SSL_ENV = "S3_VERIFY_SSL"
TRACKING_S3_FORCE_PATH_STYLE_ENV = "S3_FORCE_PATH_STYLE"


DEFAULT_REGION_ENVS = {
    SLURM_REGIONS.o2: {
        "OMPI_MCA_btl": "smcuda",
        "OMPI_MCA_pml": "ucx",
        "OMPI_MCA_osc": "ucx",
        "OMPI_MCA_coll_hcoll_enable": "0",
        "UCX_TLS": "all",
        "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "UCX_MAX_RNDV_RAILS": "4",
        "UCX_MAX_EAGER_RAILS": "4",
        "UCX_LOG_LEVEL": "ERROR",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_RNDV_SCHEME": "get_zcopy",
        "NCCL_IB_HCA": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "NCCL_IB_DISABLE": "0",
        "MOONCAKE_DEVICE": "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
        "MOONCAKE_PROTOCOL": "rdma",
        "GLOO_SOCKET_IFNAME": "ib0",
        "MLIR_ENABLE_DUMP": "0",
        "LLVM_IR_ENABLE_DUMP": "0",
        "HOME": _DEFAULT_HOME,
        "PYTORCH_KERNEL_CACHE_PATH": "/tmp/.cache",
        "TRITON_CACHE_DIR": "/tmp/.cache/triton/\\$LOCAL_RANK",
        "ENV_FILEPATH_WANDB": "/mnt/nfs/01-home/infidelis/.wandb.env",
        "WANDB_BASE_URL": "https://wb.static.space",
        "WANDB_INSECURE_DISABLE_SSL": "true",
        "WANDB_DIR": "/mnt/nfs/07-share/checkpoints/wandb",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/mnt/nfs/01-home/infidelis/.cache/benches",
        "ENV_FILEPATH_S3_DB": "/mnt/nfs/07-share/optimusprime/.config/super_secret/s3_db.env",
        "RCLONE_PATH": "/usr/bin/rclone",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
        JOB_DISPATCHER_S3_DEST_ENV: DEFAULT_JOB_DISPATCHER_SLURM_S3_DEST,
    },
    SLURM_REGIONS.o1: {
        "OMPI_MCA_btl": "smcuda",
        "OMPI_MCA_pml": "ucx",
        "OMPI_MCA_osc": "ucx",
        "OMPI_MCA_coll_hcoll_enable": "0",
        "UCX_TLS": "all",
        "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "UCX_MAX_RNDV_RAILS": "4",
        "UCX_MAX_EAGER_RAILS": "4",
        "UCX_LOG_LEVEL": "ERROR",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_RNDV_SCHEME": "get_zcopy",
        "NCCL_IB_HCA": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "NCCL_IB_DISABLE": "0",
        "MOONCAKE_DEVICE": "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
        "MOONCAKE_PROTOCOL": "rdma",
        "GLOO_SOCKET_IFNAME": "ib0",
        "MLIR_ENABLE_DUMP": "0",
        "LLVM_IR_ENABLE_DUMP": "0",
        "HOME": _DEFAULT_HOME,
        "PYTORCH_KERNEL_CACHE_PATH": "/tmp/.cache",
        "TRITON_CACHE_DIR": "/tmp/.cache/triton/\\$LOCAL_RANK",
        "ENV_FILEPATH_WANDB": "/mnt/nfs/01-home/infidelis/.wandb.env",
        "WANDB_BASE_URL": "https://wb.static.space",
        "WANDB_INSECURE_DISABLE_SSL": "true",
        "WANDB_DIR": "/mnt/nfs/07-share/checkpoints/wandb",
        "ENV_FILEPATH_S3_DB": "/mnt/nfs/07-share/optimusprime/.config/super_secret/s3_db.env",
        "RCLONE_PATH": "/mnt/nfs/01-home/infidelis/rclone-v1.68.1-linux-amd64/rclone",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/mnt/nfs/01-home/infidelis/.cache/benches",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
        JOB_DISPATCHER_S3_DEST_ENV: DEFAULT_JOB_DISPATCHER_SLURM_S3_DEST,
    },
    SLURM_REGIONS.o5: {
        "OMPI_MCA_btl": "smcuda",
        "OMPI_MCA_pml": "ucx",
        "OMPI_MCA_osc": "ucx",
        "OMPI_MCA_coll_hcoll_enable": "0",
        "UCX_TLS": "all",
        "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "UCX_MAX_RNDV_RAILS": "4",
        "UCX_MAX_EAGER_RAILS": "4",
        "UCX_LOG_LEVEL": "ERROR",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_RNDV_SCHEME": "get_zcopy",
        "NCCL_IB_HCA": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        "NCCL_IB_DISABLE": "0",
        "MOONCAKE_DEVICE": "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
        "MOONCAKE_PROTOCOL": "rdma",
        "GLOO_SOCKET_IFNAME": "ib0",
        "MLIR_ENABLE_DUMP": "0",
        "LLVM_IR_ENABLE_DUMP": "0",
        "HOME": _DEFAULT_HOME,
        "PYTORCH_KERNEL_CACHE_PATH": "/tmp/.cache",
        "TRITON_CACHE_DIR": "/tmp/.cache/triton/\\$LOCAL_RANK",
        "ENV_FILEPATH_WANDB": "/mnt/nfs/01-home/infidelis/.wandb.env",
        "WANDB_BASE_URL": "https://wb.static.space",
        "WANDB_INSECURE_DISABLE_SSL": "true",
        "WANDB_DIR": "/mnt/nfs/07-share/checkpoints/wandb",
        "ENV_FILEPATH_S3_DB": "/mnt/nfs/07-share/optimusprime/.config/super_secret/s3_db.env",
        "RCLONE_PATH": "/mnt/nfs/01-home/infidelis/rclone-v1.69.3-linux-amd64/rclone",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/mnt/nfs/01-home/infidelis/.cache/benches",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
        JOB_DISPATCHER_S3_DEST_ENV: DEFAULT_JOB_DISPATCHER_SLURM_S3_DEST,
    },
    CLOUD_REGIONS.sr004: {
        "OMPI_MCA_btl": "^openib,smcuda",
        "OMPI_MCA_pml": "ucx",
        "MOONCAKE_DEVICE": "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
        "MOONCAKE_PROTOCOL": "rdma",
        "WANDB_BASE_URL": "https://wandb.sberdevices.ru",
        "WANDB_INSECURE_DISABLE_SSL": "true",
        "WANDB_DIR": "/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/home/jovyan/.cache/benches",
        "JOB_DISPATCHER_CALLBACK_SIGNAL_FILES": "/home/jovyan/.pretrain_utils/pam-autoconfigs",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
        "ENV_FILEPATH_S3_DB": "/home/jovyan/gdedov/.config/super_secret/s3_db.env",
        **DEFAULT_JOB_DISPATCHER_S3_ENVS,
    },
    CLOUD_REGIONS.sr006: {
        "OMPI_MCA_btl": "^openib,smcuda",
        "OMPI_MCA_pml": "ucx",
        "MOONCAKE_DEVICE": "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
        "MOONCAKE_PROTOCOL": "rdma",
        "WANDB_BASE_URL": "https://wandb.sberdevices.ru",
        "WANDB_DIR": "/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/home/jovyan/.cache/benches",
        "JOB_DISPATCHER_CALLBACK_SIGNAL_FILES": "/home/jovyan/.pretrain_utils/pam-autoconfigs",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
        "ENV_FILEPATH_S3_DB": "/home/jovyan/gdedov/.config/super_secret/s3_db.env",
        **DEFAULT_JOB_DISPATCHER_S3_ENVS,
    },
    CLOUD_REGIONS.sr008: {
        "OMPI_MCA_btl": "^openib,smcuda",
        "OMPI_MCA_pml": "ucx",
        "MOONCAKE_PROTOCOL": "rdma",
        "WANDB_BASE_URL": "https://wandb.sberdevices.ru",
        "WANDB_DIR": "/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts",
        "CACHED_DATASETS_DIRECTORY_TEMPLATE": "/workspace-SR008.fs2/.cache/benches",
        **DEFAULT_JOB_DISPATCHER_API_ENVS,
        "ENV_FILEPATH_S3_DB": "/home/jovyan/gdedov/.config/super_secret/s3_db.env",
        **DEFAULT_JOB_DISPATCHER_S3_ENVS,
    },
}


# The metric sampler's raw root on each island its collector reads: a training writes <root>/<experiment>/<run>.
METRIC_SAMPLER_RAW_ROOTS = {
    SLURM_REGIONS.o1: "/mnt/nfs/07-share/optimusprime/metric-sampler/node",
    SLURM_REGIONS.o5: "/mnt/nfs/07-share/optimusprime/metric-sampler/node",
}

# jinja2 template
DEFAULT_SLURM_REGION_TEMPLATE_PARAMS = {
    SLURM_REGIONS.o1: {
        "nfs_path": "/mnt/nfs",
    },
    SLURM_REGIONS.o5: {
        "nfs_path": "/mnt/nfs",
        "default_mounts": "/mnt/nfs-extra:/mnt/nfs-extra",
    },
    SLURM_REGIONS.o2: {
        "nfs_path": "/mnt/nfs",
    },
}

# default images and log paths
DEFAULT_SLURM_REGION_PARAMS = {
    SLURM_REGIONS.o1: {
        "job_logdir": "/mnt/nfs/01-home/infidelis/jobs_logs",
        "slurm_logdir": "/mnt/nfs/01-home/infidelis/slurm_logs",
    },
    SLURM_REGIONS.o5: {
        "job_logdir": "/mnt/nfs/01-home/infidelis/jobs_logs",
        "slurm_logdir": "/mnt/nfs/01-home/infidelis/slurm_logs",
    },
    SLURM_REGIONS.o2: {
        "job_logdir": "/mnt/nfs/07-share/jobs_logs",
        "slurm_logdir": "/mnt/nfs/07-share/slurm_logs",
    },
}

# region -> workspace name (used in tracking requests)
TRACKING_WORKSPACE_MAP = {
    CLOUD_REGIONS.sr004: "pretrain",
    CLOUD_REGIONS.sr006: "pretrain",
    CLOUD_REGIONS.sr008: "pretrain",
    SLURM_REGIONS.o1: "papaya",
    SLURM_REGIONS.o2: "papaya",
    SLURM_REGIONS.o5: "papaya",
}

TRACKING_DAEMON_NAMESPACE = "tracker_daemons_v2"
TRACKING_DAEMON_INBOX_S3_KEY_TEMPLATE = (
    f"{TRACKING_DAEMON_NAMESPACE}/{{workspace}}_{{cluster}}_inbox.sqlite"
)
TRACKING_DAEMON_INBOX_S3_KEY_MAP = {
    region: TRACKING_DAEMON_INBOX_S3_KEY_TEMPLATE.format(
        workspace=workspace,
        cluster=region.lower(),
    )
    for region, workspace in TRACKING_WORKSPACE_MAP.items()
}
