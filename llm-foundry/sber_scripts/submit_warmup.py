"""Submit Triton autotune or DeepGEMM JIT warmup jobs to SLURM / cloud.

DEPRECATED: This script is superseded by the job_submitter-based workflow.
Please use ``sber_scripts/jobs/examples/warmup.py`` instead, which provides
the same functionality through the standard WarmupJobLLMF class with
built-in validation, region auto-detection, and a cleaner interface.

This file is kept for backward compatibility and will be removed in a
future release.

Two submission modes are supported (controlled by SUBMIT_GROUPS_SEPARATELY):

    True  — one job per kernel group / GEMM variant; SLURM can run them in
             parallel when GPUs are free.  Each job needs a single GPU.
    False — one job that runs all groups sequentially on a single GPU.

Usage:
    1. Fill in the settings below.
    2. python sber_scripts/submit_warmup.py
"""

import os
import sys
from pathlib import Path

# ── add helper libs to path (same pattern as submit_template.py) ─────────────
_SBER = Path(__file__).parent
sys.path.extend([
    str(_SBER / "sft" / "merger"),
    str(_SBER / "tg_utils"),
])

from local_region_configs import (   # noqa: E402
    SLURM_REGIONS,
    CLOUD_REGIONS,
    DEFAULT_REGION_ENVS,
    DEFAULT_SLURM_REGION_PARAMS,
    DEFAULT_CLOUD_REGION_IMAGES,
    get_instance_type,
)

#################################################################################
# ── USER SETTINGS ──────────────────────────────────────────────────────────────
#################################################################################

SUBMISSION_TYPE = ""       # "slurm" or "cloud"

# slurm: o3-64, o3-128, o4, o1, o5
# cloud: SR004, SR008
REGION = ""                 # specify your region here

USER_LOGIN = ""                   # your login / telegram nickname
EXP_NAME   = ""     # short label used in job description

# set config name for warmup (for example: "ultra_warmup_config.yaml")
WARMUP_CONFIG = "ultra_warmup_config.yaml"
# Which warmup to run: "triton" or "deep_gemm"
WARMUP_TYPE = "triton"          # "triton" or "deep_gemm"

# True  → one SLURM job per kernel group (groups run in parallel on the cluster)
# False → one SLURM job that runs all groups sequentially on a single GPU
SUBMIT_GROUPS_SEPARATELY = True

# ── Image / logdir overrides (None = use region defaults) ──────────────────
BASE_IMAGE  = ""
JOB_LOGDIR  = None
SLURM_LOGDIR = None

# ── cloud-only settings ─────────────────────────────────────────────────────
PRIORITY_CLASS = None    # used only for cloud
QUEUE_NAME = "shared" #"pretrain-ml" only for cloud

#################################################################################
# ── VALIDATION ─────────────────────────────────────────────────────────────────
#################################################################################

assert USER_LOGIN,  "USER_LOGIN must be set"
assert REGION,      "REGION must be set"
assert WARMUP_TYPE in ("triton", "deep_gemm"), \
    "WARMUP_TYPE must be 'triton' or 'deep_gemm'"
assert SUBMISSION_TYPE in ("slurm", "cloud"), \
    "SUBMISSION_TYPE must be 'slurm' or 'cloud'"

# ── WARMUP CONFIG ───────────────────────────────────────────────────
import yaml

CODE_PATH = str(Path(__file__).absolute().parent.parent)

warmup_config_path = f"{CODE_PATH}/sber_configs/warmup/{WARMUP_CONFIG}"

with open(warmup_config_path, "r") as f:
    warmup_config = yaml.load(f, Loader=yaml.FullLoader)[WARMUP_TYPE]

assert warmup_config["cache_dir"], f"cache_dir must be set for {WARMUP_TYPE} warmup"

CACHE_DIR = warmup_config["cache_dir"]

# ── DeepGEMM k_grouped wgrad: pipeline-stages coverage ─────────────────────
#
# The k_grouped wgrad kernel's `num_stages` template parameter depends on
# max_k (the maximum per-expert token count) via shared-memory pressure.
# On SM90 with the common FP8 1D1D block config (128x152x128, bf16 output):
#
#   max_k <= 127 744  ->  num_stages = 5
#   max_k >  127 744  ->  num_stages = 4
#
# Each distinct num_stages produces a different kernel hash.  If training
# encounters a max_k above the threshold but the warmup never reached it,
# the kernel is recompiled from scratch (slow!).
#
# Fix: max_size must exceed 127 744 so the sweep's group distributions
# eventually produce a max_k that crosses the boundary.  When raising it
# would OOM (max_size * num_groups is huge), a separate "stages-coverage"
# pass is added with --max-size-override and --max-buffer-tokens.
_SM90_STAGES_THRESHOLD = 128_000  # just above 127 744
_GPU_MEM_BUDGET = 70 * (1 << 30)  # conservative 70 GiB for 80 GB GPU
_NEED_STAGES_PASS = False
_STAGES_EXTRA_ARGS = ""

if WARMUP_TYPE == "deep_gemm":
    _dg_max_size = warmup_config.get("max_size", 8192)
    _dg_num_groups = warmup_config.get("num_groups", 1)
    _dg_hid_dim = warmup_config.get("hid_dim", 7168)
    _dg_int_dim = warmup_config.get("int_dim", 2048)

    if _dg_max_size < _SM90_STAGES_THRESHOLD:
        _largest_dim = max(_dg_hid_dim, 2 * _dg_int_dim)
        _buf_tokens = _SM90_STAGES_THRESHOLD * _dg_num_groups
        _peak_est = _buf_tokens * _largest_dim * 2 * 3

        if _peak_est < _GPU_MEM_BUDGET:
            print(f"[stages-cov] Will use --max-size-override {_SM90_STAGES_THRESHOLD} "
                  f"(est. peak ~ {_peak_est / (1 << 30):.1f} GiB)")
            _STAGES_EXTRA_ARGS = f" --max-size-override {_SM90_STAGES_THRESHOLD}"
        else:
            _NEED_STAGES_PASS = True
            _stages_buf = min(
                _SM90_STAGES_THRESHOLD * 2,
                _GPU_MEM_BUDGET // (_largest_dim * 2 * 3),
            )
            _stages_buf = (_stages_buf // 128) * 128
            _STAGES_EXTRA_ARGS = (
                f" --max-size-override {_SM90_STAGES_THRESHOLD}"
                f" --max-buffer-tokens {_stages_buf}"
            )
            print(f"[stages-cov] Full raise needs ~ {_peak_est / (1 << 30):.1f} GiB; "
                  f"adding separate stages-coverage pass"
                  f" (--max-buffer-tokens {_stages_buf}).")

#################################################################################
# ── PATHS ──────────────────────────────────────────────────────────────────────
#################################################################################

_OPS_WARMUP  = f"{CODE_PATH}/llmfoundry/models/ops/warmup"

external_code_libraries = [
    CODE_PATH,
    f"{CODE_PATH}/contrib/composer",
    f"{CODE_PATH}/contrib/streaming",
    f"{CODE_PATH}/contrib/gigafsdp",
    f"{CODE_PATH}/contrib/metric-sampler",
]

for _lib in external_code_libraries:
    assert os.path.exists(_lib) and os.listdir(_lib), \
        f"Code directory does not exist or is empty: {_lib}"

#################################################################################
# ── RESOLVE IMAGE / LOGDIRS ────────────────────────────────────────────────────
#################################################################################

if SUBMISSION_TYPE == "slurm":
    _region_params = DEFAULT_SLURM_REGION_PARAMS.get(REGION, {})
    if _region_params:
        JOB_LOGDIR   = JOB_LOGDIR   or _region_params.get("job_logdir")
        SLURM_LOGDIR = SLURM_LOGDIR or _region_params.get("slurm_logdir")
        BASE_IMAGE   = BASE_IMAGE   or _region_params.get("default_image_path")
    else:
        assert JOB_LOGDIR and SLURM_LOGDIR and BASE_IMAGE, (
            "Custom SLURM region: please set JOB_LOGDIR, SLURM_LOGDIR, BASE_IMAGE."
        )
else:  # cloud
    assert REGION in DEFAULT_CLOUD_REGION_IMAGES, \
        f"Unknown cloud region: {REGION}"
    BASE_IMAGE = BASE_IMAGE or DEFAULT_CLOUD_REGION_IMAGES[REGION]

assert BASE_IMAGE, "BASE_IMAGE must be resolved — check your REGION setting."

PARTITION = "main" if PRIORITY_CLASS == "high" else "middle"

#################################################################################
# ── ENVIRONMENT VARIABLES ──────────────────────────────────────────────────────
#################################################################################

env_variables = {
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
    "PYTHONPATH": ":".join(external_code_libraries),
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    **DEFAULT_REGION_ENVS.get(REGION, {}),
}

if WARMUP_TYPE == "triton":
    # Make Triton write its cache to our shared directory.
    # env_variables["TRITON_CACHE_DIR"] = CACHE_DIR
    env_variables["TRITON_CONFIGS_DIR"] = CACHE_DIR
    env_variables["TRITON_MODE"] = "warmup"

elif WARMUP_TYPE == "deep_gemm":
        env_variables["DG_JIT_CACHE_DIR"] = CACHE_DIR
        os.makedirs(os.path.join(CACHE_DIR, "tmp"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "cache"), exist_ok=True)

#################################################################################
# ── BUILD JOB LIST ─────────────────────────────────────────────────────────────
#################################################################################

# Each entry: (job_label, shell_script_string)
kernel_groups = []
jobs = []

if WARMUP_TYPE == "triton":
    kernel_groups.extend(warmup_config.get("kernel_groups")  or [])
    kernel_groups.extend(warmup_config.get("dense_groups")  or [])
    assert kernel_groups, "Warmup config must contain at least one kernel group"
    _base_cmd = (
        f"python {_OPS_WARMUP}/triton_warmup.py"
        f" --config {warmup_config_path}"
        f" --device 0"
    )
    if SUBMIT_GROUPS_SEPARATELY:
        for group in kernel_groups:
            jobs.append((
                f"triton_{group}",
                f"{_base_cmd} --kernel-group {group}",
            ))
    else:
        # No --kernel-group → runs all groups sequentially.
        jobs.append(("triton_all", _base_cmd))

elif WARMUP_TYPE == "deep_gemm":
    kernel_groups.extend(warmup_config.get("gemm_groups")  or [])
    assert kernel_groups, "Warmup config must contain at least one kernel group"
    _base_cmd = (
        f"python {_OPS_WARMUP}/deep_gemm_warmup.py"
        f" --config {warmup_config_path}"
        f" --device 0"
    )
    if SUBMIT_GROUPS_SEPARATELY:
        for gemm in kernel_groups:
            jobs.append((
                f"deepgemm_{gemm}",
                f"{_base_cmd} --gemm {gemm}",
            ))
    else:
        jobs.append(("deepgemm_all", _base_cmd))

    # Stages-coverage: either inline override or separate pass
    if _STAGES_EXTRA_ARGS:
        _stages_cmd = _base_cmd + _STAGES_EXTRA_ARGS
        if _NEED_STAGES_PASS:
            if SUBMIT_GROUPS_SEPARATELY:
                for gemm in kernel_groups:
                    jobs.append((
                        f"deepgemm_stages_{gemm}",
                        f"{_stages_cmd} --gemm {gemm}",
                    ))
            else:
                jobs.append(("deepgemm_stages_all", _stages_cmd))
        else:
            # Memory fits: just re-run the same jobs with the override
            if SUBMIT_GROUPS_SEPARATELY:
                for gemm in kernel_groups:
                    jobs.append((
                        f"deepgemm_stages_{gemm}",
                        f"{_stages_cmd} --gemm {gemm}",
                    ))
            else:
                jobs.append(("deepgemm_stages_all", _stages_cmd))

#################################################################################
# ── SUBMIT ─────────────────────────────────────────────────────────────────────
#################################################################################

print(f"Submitting {len(jobs)} job(s)  [type={WARMUP_TYPE}  region={REGION}  mode={'per-group' if SUBMIT_GROUPS_SEPARATELY else 'sequential'}]")

if SUBMISSION_TYPE == "slurm":
    from job_templates import submit_slurm_job  # noqa: E402

    for job_label, script in jobs:
        print(f"\n  [{job_label}] {script}")
        submit_slurm_job(
            job_desc=f"{USER_LOGIN} | {EXP_NAME} | {job_label}",
            base_image=BASE_IMAGE,
            n_workers=1,
            gpu_num=1,
            type="binary",
            processes_per_worker=1,
            region=REGION,
            script=script,
            env_variables=env_variables,
            job_logdir=JOB_LOGDIR,
            slurm_logdir=SLURM_LOGDIR,
            max_restarts=0,
            nodelist=None,
            partition=PARTITION,
        )
raise Exception(f"Please use job_submitter {Path(__file__).parent / 'jobs/examples/'} to submit jobs. This script is deprecated.")
else:  # cloud
    import client_lib  # noqa: E402

    _instance_type = get_instance_type(REGION, 1)
    for job_label, script in jobs:
        print(f"\n  [{job_label}] {script}")
        job = client_lib.Job(
            job_desc=f"{USER_LOGIN} | {EXP_NAME} | {job_label}",
            base_image=BASE_IMAGE,
            script=script,
            n_workers=1,
            processes_per_worker=1,
            instance_type=_instance_type,
            type="binary",
            preflight_check=False,
            env_variables=env_variables,
            region=REGION,
            pytorch_use_env=True,
            max_retry=None,
            priority_class=PRIORITY_CLASS,
            queue_name=QUEUE_NAME,
            flags={}
        )
        print(job.submit())
