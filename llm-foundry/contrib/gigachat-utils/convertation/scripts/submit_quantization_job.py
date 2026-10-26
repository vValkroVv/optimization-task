#!/usr/bin/env python3

import os
from pathlib import Path
import client_lib
import sys


def update_sys_path(paths) -> None:
    sys.path.extend([str(Path(path).absolute()) for path in paths])

update_sys_path([
    Path(__file__).parent.parent.parent / "job_templates/src",
])

from job_templates import (
    get_instance_type
)

# constants setting ---------------------------------------------
TELEGRAM_LOGIN = 
REGION =  # "SR008" or "SR004"

N_GPUS = 1 # fix it if you need more ram
N_WORKERS = 1

# launch block -------------------------------------------------

INSTANCE_TYPE = get_instance_type(REGION, N_GPUS)

BASE_IMAGE = "cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/llm-foundry-train:2025_09_15_commit7bddfa74"

CODE_PATH = str(Path(__file__).absolute().parent.parent)

external_code_libraries = [
    CODE_PATH,
]

assert N_WORKERS == 1, "Quantization is single node job"
for path in external_code_libraries:
    assert (
        os.path.exists(path) and len(os.listdir(path)) > 0
    ), f"Code directory {path} do not exist or empty!"

quantization_script_path = f"{CODE_PATH}/quantization/fine_grained_fp8_quantization.py"
script = f"python {quantization_script_path}"

env_variables = {
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python", 
    "PYTHONPATH": ":".join(external_code_libraries),
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'TORCH_NCCL_AVOID_RECORD_STREAMS': '1',
    'NCCL_DEBUG': 'WARN',
    "NCCL_DEBUG_SUBSYS": "GRAPH",
    'OMPI_MCA_coll_hcoll_enable': '0',
    "TORCHINDUCTOR_COMPILE_THREADS": "8",
    'OMPI_MCA_btl': '^openib,smcuda',
    'OMPI_MCA_pml': 'ucx',
    'OMP_NUM_THREADS': '32',
}

print(f"Submitting job for {N_WORKERS * N_GPUS} GPUs with the following script: {script}")
job = client_lib.Job(
    job_desc=f"{TELEGRAM_LOGIN} | quantization ckpt job",
    script=script,
    base_image=BASE_IMAGE,
    n_workers=N_WORKERS,
    processes_per_worker=1,
    instance_type=INSTANCE_TYPE,
    type="binary",
    preflight_check=False,
    env_variables=env_variables,
    region=REGION,
    pytorch_use_env=True,
    max_retry=None,
    flags={},
)
print(job.submit())
