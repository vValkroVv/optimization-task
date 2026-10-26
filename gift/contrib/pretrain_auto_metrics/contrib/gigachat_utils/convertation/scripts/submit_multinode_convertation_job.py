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
N_GPUS_PER_JOB = 1     # сколько GPU бронировать на ноду (для памяти)
WORLD_SIZE = 1         # <- СКОЛЬКО РАНГОВ/НОД ХОТИМ

REGION = None # "SR004"
INSTANCE_TYPE = None # f"a100.{N_GPUS}gpu"
BASE_IMAGE = None # "path_or_link_to_container_image"

CONFIG_NAME = None # "your_config_name.yaml"

CODE_PATH = str(Path(__file__).absolute().parent.parent)

# launch block -------------------------------------------------

CONVERT_SCRIPT = f"{CODE_PATH}/map-fsdp-to-fsdp.py"
CONFIG_PATH = f"{CODE_PATH}/configs/{CONFIG_NAME}"


external_code_libraries = [CODE_PATH]
if LLM_FOUNDRY_PATH:
    external_code_libraries.append(f"{LLM_FOUNDRY_PATH}/contrib/gigafsdp")

for path in external_code_libraries:
    assert os.path.exists(path) and os.listdir(path), f"Code directory {path} does not exist or is empty!"

print(f"Submitting {WORLD_SIZE} single-node jobs (1 process each)")

job_ids = []
for rank in range(WORLD_SIZE):
    script = (
        f"python {CONVERT_SCRIPT} --input-config-path {CONFIG_PATH} "
        f"--global-rank {rank} --world-size {WORLD_SIZE}"
    )

    env = {
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        "PYTHONPATH": ":".join(external_code_libraries),
        "RANK": str(rank),
        "WORLD_SIZE": str(WORLD_SIZE),
    }

    job = client_lib.Job(
        job_desc=f"{TELEGRAM_LOGIN} | ckpt convert rank={rank}/{WORLD_SIZE}",
        script=script,
        base_image=BASE_IMAGE,
        n_workers=1,
        processes_per_worker=1,
        instance_type=INSTANCE_TYPE,
        type="binary",
        preflight_check=False,
        env_variables=env,
        region=REGION,
        pytorch_use_env=True,
        max_retry=None,
        flags={},
    )
    job_id = job.submit()
    job_ids.append(job_id)
    print(f"submitted rank {rank}: {job_id}")

print("All jobs submitted:", job_ids)
