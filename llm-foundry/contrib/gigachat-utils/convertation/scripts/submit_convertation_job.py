#!/usr/bin/env python3

import os
from pathlib import Path
import client_lib

warning_msg = (
    "This submit script is quite deprecated and needs to be updated to work properly. "
    "Please use submit template in llm-foundry with convertation job type or "
    "update all the params to proper values (config name, image path, region and istance type)."
)
raise RuntimeError(warning_msg)

# --- Constants setting ---
TELEGRAM_LOGIN = None
LLM_FOUNDRY_PATH = ""

# Increase if you need more RAM
N_GPUS = 1
N_WORKERS = 1

REGION = None # "SR004"
INSTANCE_TYPE = None # f"a100.{N_GPUS}gpu"
BASE_IMAGE = None # "path_or_link_to_container_image"

CONFIG_NAME = None # "your_config_name.yaml"

CODE_PATH = str(Path(__file__).absolute().parent.parent)

# launch block -------------------------------------------------

external_code_libraries = [
    CODE_PATH,
    f"{LLM_FOUNDRY_PATH}/contrib/gigafsdp"
]

N_GPUS = min(8, N_GPUS)
assert N_WORKERS == 1, "Convertation is single node job"
assert TELEGRAM_LOGIN is not None or TELEGRAM_LOGIN != "", f"Please, provide TG login! Got {TELEGRAM_LOGIN}."

for path in external_code_libraries:
    assert (
        os.path.exists(path) and len(os.listdir(path)) > 0
    ), f"Code directory {path} do not exist or empty!"

convertation_script_path = f"{CODE_PATH}/map-fsdp-to-fsdp.py"
config_path = f"{CODE_PATH}/configs/{CONFIG_NAME}"
script = f"python {convertation_script_path} --input-config-path {config_path}"

print(f"Submitting job for {N_WORKERS * N_GPUS} GPUs with the following script: {script}")
job = client_lib.Job(
    job_desc=f"{TELEGRAM_LOGIN} | convertation ckpt job",
    script=script,
    base_image=BASE_IMAGE,
    n_workers=N_WORKERS,
    processes_per_worker=1,
    instance_type=INSTANCE_TYPE,
    type="binary",
    preflight_check=False,
    env_variables={
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        "PYTHONPATH": ":".join(external_code_libraries),
    },
    region=REGION,
    pytorch_use_env=True,
    max_retry=None,
    flags={},
)
print(job.submit())
