import datetime
import os
import sys
from pathlib import Path
from typing import List
import warnings


#####################################
# import configs
def update_sys_path(paths: List[str]) -> None:
    sys.path.extend([str(Path(path).absolute()) for path in paths])

update_sys_path([
    Path(__file__).parent / 'sft' / 'merger',
    Path(__file__).parent / 'tg_utils',
])

from local_region_configs import (
    SLURM_REGIONS,
    CLOUD_REGIONS,
    DEFAULT_REGION_ENVS,
    DEFAULT_SLURM_REGION_PARAMS,
    DEFAULT_CLOUD_REGION_IMAGES,
    get_instance_type
)

from config_merger import compile_config

try:
    from tracker_utils import auto_track_from_config
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    warnings.warn(
        "tracker_utils module not found. Tracking functionality will be disabled. "
        "Auto tracking are not available.",
        ImportWarning
    )


print("CLOUD_REGIONS: ", CLOUD_REGIONS.get_region_list())
print("SLURM_REGIONS: ", SLURM_REGIONS.get_region_list())

#####################################
# constants setting ---------------------------------------------
USE_CLEARML = False
USE_SHARP = False

# Common settings
SUBMISSION_TYPE =           # "slurm" or "cloud"

# slurm: o3-64, o3-128, o4, o1, o5
# cloud: SR004, SR008
REGION =                    # specify default region here or your custom
INSTANCE_TYPE = None         # used only for cloud
PRIORITY_CLASS = "medium"    # used only for cloud
QUEUE_NAME = "shared"        # used only for cloud

# MUST-HAVE FOR CLOUD Your Telegram Nickname
USER_LOGIN =              # MUST HAVE IF YOU USE EXPERIMENT AUTO TRACKING
# MUST-HAVE FOR SLURM Your papaya codename
CODE_NAME =               # ONLY USE YOUR O CODENAME IF PAPAYA, DON'T USE USER LOGIN !!!!!
EXP_NAME =                 # 'MyFancyModel'
CONFIG_NAME =             # 'example.yaml', for inf job could be any str

# Default configs are used only for SFT and Vision training and contain parameters
# that are rarely modified. If you want to start pretraining, simply do not change
# DEFAULT_CONFIG_NAME default value
DEFAULT_CONFIG_NAME = ""     # 'example.yaml'

MAX_RETRY = None             # no retry by default
N_GPUS = 8
N_WORKERS =

BASE_IMAGE = None            # specify base image or use default
SCRIPT_TYPE = "pytorch2"     # "pytorch2", "infinity_job" or "convertation_job"
SCRIPT_ARGS = ""             # args to be passed in train.py script

# Slurm-specific settings, you need to specify them in custom regions
JOB_LOGDIR = None
SLURM_LOGDIR = None

# SETTING FOR AUTO TRACKING EXPERIMENT IN BOT
AUTO_TRACK = False


PARTITION = "main" if PRIORITY_CLASS == "high" else "middle"

ALLOCATION = "alloc-sberdevices-giga-vision-h100-sr008" if REGION == "SR008" else None

#####################################
# Validation
assert USER_LOGIN, 'USER_LOGIN is None'
assert EXP_NAME, 'EXP_NAME is None'
assert SUBMISSION_TYPE in ["slurm", "cloud"], 'SUBMISSION_TYPE must be "slurm" or "cloud"'
assert SCRIPT_TYPE in ["pytorch2", "infinity_job", "convertation_job"], 'SCRIPT_TYPE must be "pytorch2", "infinity_job" or "convertation_job"'
assert PRIORITY_CLASS in ["low", "medium", "high"], 'please use "low", "medium" or "high" for PRIORITY_CLASS'
assert PARTITION in ["middle", "main"], 'please use "middle" or "main" for PARTITION'
if PARTITION == "middle" and N_WORKERS > 32:
    assert False, "WARNING: you are submitting big job to middle partition, please consider using high PRIORITY_CLASS"

if AUTO_TRACK and SCRIPT_TYPE == "pytorch2":
    if SUBMISSION_TYPE == "slurm":
        assert CODE_NAME, 'CODE_NAME must be set for SLURM auto-tracking'
    else:
        assert USER_LOGIN, 'USER_LOGIN must be set for Cloud auto-tracking'

#####################################
# Common code libraries
CODE_PATH = str(Path(__file__).absolute().parent.parent)
print(f"{CODE_PATH =}")

external_code_libraries = [
    CODE_PATH,
    f"{CODE_PATH}/contrib/composer",
    f"{CODE_PATH}/contrib/streaming",
    f"{CODE_PATH}/contrib/gigafsdp",
    f"{CODE_PATH}/contrib/flash-linear-attention",
    f"{CODE_PATH}/contrib/metric-sampler",
]

for path in external_code_libraries:
    assert (
        os.path.exists(path) and len(os.listdir(path)) > 0
    ), f"Code directory {path} do not exist or empty!"

#####################################
# Set up paths based on submission type

N_PROCESSES_PER_WORKER = None # None or User value; # NOTE: User value affects only "cloud"

DEFAULT_IMAGE_PATH = None
if SUBMISSION_TYPE == "slurm":
    region_params = DEFAULT_SLURM_REGION_PARAMS.get(REGION, {})
    if len(region_params) != 0:
        JOB_LOGDIR = region_params.get('job_logdir') if not JOB_LOGDIR else JOB_LOGDIR
        SLURM_LOGDIR = region_params.get('slurm_logdir') if not SLURM_LOGDIR else SLURM_LOGDIR
        DEFAULT_IMAGE_PATH = region_params.get('default_image_path') if not DEFAULT_IMAGE_PATH else DEFAULT_IMAGE_PATH
    elif any((JOB_LOGDIR is None, SLURM_LOGDIR is None, BASE_IMAGE is None)):
        raise NotImplementedError((
            "Invalid REGION for SLURM submission. "
            "You must specify you own JOB_LOGDIR, SLURM_LOGDIR "
            "and BASE_IMAGE is you are using custom region"
        ))

    N_PROCESSES_PER_WORKER = 1
else:  # cloud
    assert REGION in DEFAULT_CLOUD_REGION_IMAGES
    DEFAULT_IMAGE_PATH = DEFAULT_CLOUD_REGION_IMAGES[REGION]

    if not INSTANCE_TYPE:
        INSTANCE_TYPE = get_instance_type(REGION, N_GPUS)

        if INSTANCE_TYPE is None:
            raise NotImplementedError(f"Region type '{REGION}' in not supported")

    if N_PROCESSES_PER_WORKER is None:
        N_PROCESSES_PER_WORKER = N_GPUS

BASE_IMAGE = BASE_IMAGE or DEFAULT_IMAGE_PATH
assert BASE_IMAGE is not None,  "image must be specified"

#####################################
# Base environment variables
if REGION not in DEFAULT_REGION_ENVS:
    print('Using custom region, default region env variables are empty')

env_variables = {
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
    "PYTHONPATH": ":".join(external_code_libraries),
    # "TORCH_NCCL_ENABLE_MONITORING": "1",
    # "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "30",
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'TORCH_NCCL_AVOID_RECORD_STREAMS': '1',
    'NCCL_DEBUG': 'WARN',
    "NCCL_DEBUG_SUBSYS": "GRAPH",
    # sharp off
    'OMPI_MCA_coll_hcoll_enable': '0',
    "TORCHINDUCTOR_COMPILE_THREADS": "8",
    **DEFAULT_REGION_ENVS.get(REGION, {}),
}
if USE_SHARP:
    env_variables.update({
        # 'OMPI_MCA_coll_hcoll_enable': '1',
        # 'SHARP_SMX_SOCK_ADDR_FAMILY': 'ipv4',
        'NCCL_COLLNET_ENABLE': '1',
        # 'HCOLL_ENABLE_SHARP': '1',
        # 'SHARP_COLL_ENABLE_SAT': '1',
    })

#####################################
# Prepare job
if SCRIPT_TYPE == "pytorch2":
    assert CONFIG_NAME, 'CONFIG_NAME with training details is None'
    # start llm foundry run
    if DEFAULT_CONFIG_NAME:
        is_text_dpo = Path(f"{CODE_PATH}/sber_configs/dpo/default_configs/{DEFAULT_CONFIG_NAME}").exists()
        is_text_sft = Path(f"{CODE_PATH}/sber_configs/sft/default_configs/{DEFAULT_CONFIG_NAME}").exists()
        is_vision = Path(f"{CODE_PATH}/sber_configs/vision/default_configs/{DEFAULT_CONFIG_NAME}").exists()
        is_video = Path(f"{CODE_PATH}/sber_configs/video/default_configs/{DEFAULT_CONFIG_NAME}").exists()
        is_tts = Path(f"{CODE_PATH}/sber_configs/tts/default_configs/{DEFAULT_CONFIG_NAME}").exists()

        if is_text_sft:
            modality = "sft"
        elif is_vision:
            modality = "vision"
        elif is_text_dpo:
            modality = "dpo"
        elif is_tts:
            modality = "tts"
        elif is_video:
            modality = "video"
        else:
            modality = None

        assert modality is not None, "Default config is provided, but does not exists in ./sber_configs/sft/default_configs/ or ./sber_configs/vision/default_configs/ or ./sber_configs/dpo/default_configs/"
        assert int(is_text_sft) + int(is_vision) + int(is_video) + int(is_text_dpo) + int(is_tts) == 1, "Default config exists in both folders, make sure it has unique name."

        default_config_path = f"{CODE_PATH}/sber_configs/{modality}/default_configs/{DEFAULT_CONFIG_NAME}"
        editable_config_path = f"{CODE_PATH}/sber_configs/{modality}/editable_configs/{CONFIG_NAME}"
        merged_config_path = f"{CODE_PATH}/sber_configs/{modality}/merged_configs"

        assert os.path.exists(default_config_path), f"Provided default config does not exist: {default_config_path}"
        assert os.path.exists(editable_config_path), f"Provided config does not exist: {editable_config_path}"

        curr_date = datetime.datetime.today().strftime('%m-%d-%Y')
        config_path = f"{merged_config_path}/merged-{curr_date}-{DEFAULT_CONFIG_NAME.replace('.yaml', '')}_{CONFIG_NAME.replace('.yaml', '')}.yaml"
        config_path, config_info = compile_config(editable_config_path, default_config_path, region=REGION,
                                                  save_path=config_path)
        import yaml
        with open(config_path, 'r') as f:
            merged_config_dict = yaml.safe_load(f)
        gbs, tp_size, sp_size = config_info['global_train_batch_size'], config_info['tp_size'], config_info['sp_size']

        if isinstance(sp_size, int):
            sp_size = [{'sp_size': sp_size}]
        for sp_cfg_ in sp_size:
            _sp_size = sp_cfg_['sp_size']
            assert gbs % (N_GPUS * N_WORKERS // (tp_size * _sp_size)) == 0, \
                (
                    f'gbs mod N_GPUS * N_WORKERS must be 0, '
                    f'gpus={N_GPUS}*{N_WORKERS} gbs={gbs} '
                    f'tp_size={tp_size} sp_size={_sp_size}'
                )

        # update exp name
        EXP_NAME = f"{config_info['exp_name']} | {EXP_NAME}"
    else:
        config_path = f"{CODE_PATH}/sber_configs/{CONFIG_NAME}"

    assert os.path.exists(config_path), f"Provided config does not exist: {config_path}"
    train_script_path = f"{CODE_PATH}/scripts/train/train.py"
    script = f"{train_script_path} {config_path} use_clearml={str(USE_CLEARML).lower()} {SCRIPT_ARGS}"

elif SCRIPT_TYPE == "infinity_job":
    script = f"sleep infinity"
    SCRIPT_TYPE = "binary"
elif SCRIPT_TYPE=="convertation_job":
    assert N_WORKERS == 1, "Convertation is single node job"
    for path in external_code_libraries:
        assert (
            os.path.exists(path) and len(os.listdir(path)) > 0
        ), f"Code directory {path} do not exist or empty!"

    convertation_script_path = f"{CODE_PATH}/contrib/gigachat-utils/convertation/map-fsdp-to-fsdp.py"
    config_path = f"{CODE_PATH}/sber_configs/convertation/{CONFIG_NAME}"
    script = f"python {convertation_script_path} --input-config-path {config_path}"
    SCRIPT_TYPE = "binary"

    N_PROCESSES_PER_WORKER = 1

else:
    raise NotImplementedError(f"Script type '{SCRIPT_TYPE}' is not supported")

if AUTO_TRACK and TRACKER_AVAILABLE and SCRIPT_TYPE == "pytorch2":
    run_auto_track(
        region=REGION,
        submission_type=SUBMISSION_TYPE,
        code_name=CODE_NAME,
        user_login=USER_LOGIN,
        config_path=config_path,
        config_name=CONFIG_NAME,
        default_config_name=DEFAULT_CONFIG_NAME,
        merged_config_dict=merged_config_dict
    )


#####################################
# Submit job
print(f"Submitting job for {N_WORKERS * N_GPUS} GPUs with the following script:")
print(script)
raise Exception(f"Please use job_submitter {Path(__file__).parent / 'jobs/examples/'} to submit jobs. This script is deprecated.")
if SUBMISSION_TYPE == "slurm":
    from job_templates import submit_slurm_job

    submit_slurm_job(
        job_desc=f"{USER_LOGIN} | {EXP_NAME}",
        base_image=BASE_IMAGE,
        n_workers=N_WORKERS,
        gpu_num=N_GPUS,
        type=SCRIPT_TYPE,
        processes_per_worker=N_PROCESSES_PER_WORKER,
        region=REGION,
        script=script,
        env_variables=env_variables,
        job_logdir=JOB_LOGDIR,
        slurm_logdir=SLURM_LOGDIR,
        max_restarts=MAX_RETRY,
        nodelist=None,
        partition=PARTITION,
    )
else:  # cloud
    import client_lib

    job = client_lib.Job(
        job_desc=f"{USER_LOGIN} | {EXP_NAME} | #gigachat #prod |",
        base_image=BASE_IMAGE,
        script=script,
        n_workers=N_WORKERS,
        processes_per_worker=N_PROCESSES_PER_WORKER,
        instance_type=INSTANCE_TYPE,
        type=SCRIPT_TYPE,
        preflight_check=True,
        env_variables=env_variables,
        region=REGION,
        pytorch_use_env=True,
        max_retry=MAX_RETRY,
        priority_class=PRIORITY_CLASS,
        flags={},
        # allocation_name=ALLOCATION,
        queue_name=QUEUE_NAME,
    )
    print(job.submit())

    print("Wait before requesting logs")
    # import time
    # time.sleep(15)
    # job.logs(verbose=False)
