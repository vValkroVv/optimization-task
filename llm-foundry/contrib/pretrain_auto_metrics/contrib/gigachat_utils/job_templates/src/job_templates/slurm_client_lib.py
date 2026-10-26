import getpass
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from jinja2 import Template

from job_templates.region_configs import DEFAULT_SLURM_REGION_TEMPLATE_PARAMS

_PYTORCH2_SCRIPT = """\
python -m torch.distributed.launch \
    --nproc-per-node=\\$GPUS_PER_NODE \
    --nnodes=\\$NNODES \
    --node-rank=\\$SLURM_PROCID \
    --master-addr=\\$SLURM_LAUNCH_NODE_IPADDR \
    --master-port=12345 \
    --use-env \
    {script}
"""


class SlurmRESTError(RuntimeError):
    pass


def _submit_sbatch(script_path, job_logdir, slurm_logdir):
    try:
        print(f"Submitting sbatch with script: {script_path}")
        result = subprocess.run(
            ["sbatch", script_path], capture_output=True, text=True, check=True
        )

        print("Submission output:", result.stdout)
        print("Submission error (if any):", result.stderr)

        if not result.stderr:
            job_id = result.stdout.split(" ")[-1].strip()
            print(
                f"Job logdir: `{job_logdir}/{job_id}`\n"
                f"Slurm logdir: `{slurm_logdir}/{job_id}`"
            )
            print(
                "You can try use commands (if they specified in your bashrc file):\n"
                f"    `slurm_logs {job_id}` - check slurm logs dir\n"
                f"    `job_logs {job_id}` - check rank 0 logs of job\n"
            )
            print(
                "If you created infinity job, you can attach to it with command (use **tmux** or **screen** for this):\n"
                f"    `srun --pty --overlap --jobid {job_id} bash`"
            )
        return job_id

    except subprocess.CalledProcessError as e:
        print("Error while submitting job:")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error message:", e.stderr)
        raise


def _submit_slurmrestd(
    slurmrestd_urls: list[str],
    slurm_jwt_token: str,
    script: str,
    slurm_logdir: str,
    job_logdir: str,
    job_desc: str,
    n_workers: int,
    gpu_num: int,
    processes_per_worker: int,
    max_restarts: Optional[str] = None,
    nodelist: Optional[str] = None,
    partition: str = "middle",
):
    urls = [f"{url.rstrip('/')}/slurm/v0.0.40/job/submit" for url in slurmrestd_urls]
    headers = {
        "Content-Type": "application/json",
        "X-SLURM-USER-TOKEN": slurm_jwt_token,
    }
    payload = {
        "script": script,
        "job": {
            "name": job_desc,
            "nodes": n_workers,
            "standard_output": f"{slurm_logdir}/%j/logs.log",
            "standard_error": f"{slurm_logdir}/%j/logs.err",
            "tres_per_node": f"gres/gpu:{gpu_num}",
            "tasks_per_node": processes_per_worker,
            "current_working_directory": str(Path(__file__).parent),
            "environment": {"USER": getpass.getuser()},
            "partition": partition,
            "exclusive": True,
        },
    }
    if max_restarts:
        payload["job"]["max_restarts"] = max_restarts
        payload["job"]["requeue"] = True
    if nodelist:
        payload["job"]["nodelist"] = nodelist

    if not urls:
        raise SlurmRESTError("No slurmrestd urls provided")

    for url in urls:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                raise SlurmRESTError(f"Submit failed {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"Error submitting job to {url}: {e}")
            continue
        else:
            break
    else:
        raise SlurmRESTError("Failed to submit job to any slurmrestd url")

    data = resp.json()
    job_id = data.get("job_id")
    if not job_id:
        raise SlurmRESTError(f"Unexpected submit response: {data}")

    print(
        f"Job logdir: `{job_logdir}/{job_id}`\nSlurm logdir: `{slurm_logdir}/{job_id}`"
    )
    print(
        "You can try use commands (if they specified in your bashrc file):\n"
        f"    `slurm_logs {job_id}` - check slurm logs dir\n"
        f"    `job_logs {job_id}` - check rank 0 logs of job\n"
    )
    print(
        "If you created infinity job, you can attach to it with command (use **tmux** or **screen** for this):\n"
        f"    `srun --pty --overlap --jobid {job_id} bash`"
    )

    return job_id


def submit_slurm_job(
    base_image,
    n_workers,
    script,
    job_desc,
    region,
    job_logdir,
    slurm_logdir,
    gpu_num=8,
    processes_per_worker=1,
    nodelist=None,
    env_variables=None,
    type="pytorch2",
    max_restarts=None,
    dry_run=False,
    slurm_submit_type: str = "sbatch",
    slurmrestd_urls: list[str] = [],
    slurm_jwt_token: Optional[str] = None,
    partition: str = "middle",
    # Ray cluster specific parameters
    extra_mounts: Optional[str] = None,
    cpus_per_task: Optional[int] = None,
    **override_kwargs,
):
    if region not in DEFAULT_SLURM_REGION_TEMPLATE_PARAMS:
        sample_value = next(iter(DEFAULT_SLURM_REGION_TEMPLATE_PARAMS.values()))
        assert all(k in override_kwargs for k in sample_value.keys()), (
            f"All custom region params must be specified: {sample_value.keys()}"
        )
    assert type in ("pytorch2", "binary"), "type must be either 'pytorch2' or 'binary'"
    assert job_desc is not None, "job description cannot be empty"

    if max_restarts is not None:
        assert max_restarts > 1, "max restarts must be None or greater than 1"
        override_kwargs["max_restarts"] = max_restarts - 1

    if type == "pytorch2":
        script = _PYTORCH2_SCRIPT.format(script=script)

    if partition == "middle" and n_workers > 32:
        raise ValueError("You are submitting big job to middle partition, please consider using main partition")

    if extra_mounts:
        for mount in extra_mounts.split(','):
            mount = mount.strip()
            if not mount: # Skip empty strings (handles trailing commas)
                continue
            assert ":" in mount, (
                f"Invalid mount format: '{mount}'. "
                f"Each mount in extra_mounts must be in 'source:target' format (contains ':'). "
                f"Got full string: '{extra_mounts}'"
            )

    context = {
        "job_desc": job_desc,
        "n_workers": n_workers,
        "processes_per_worker": processes_per_worker,
        "gpu_num": gpu_num,
        "base_image": base_image,
        "script": script,
        "job_logdir": job_logdir,
        "slurm_logdir": slurm_logdir,
        "nodelist": nodelist,
        "env_variables": env_variables or {},
        "partition": partition,
        # Ray cluster specific parameters
        "extra_mounts": extra_mounts,
        "cpus_per_task": cpus_per_task,
        **DEFAULT_SLURM_REGION_TEMPLATE_PARAMS.get(region, {}),
        **override_kwargs,
    }

    with open(Path(__file__).parent / "binary.sh.j2", "r") as file:
        template_content = file.read()
    template = Template(template_content)
    output = template.render(context)

    now = datetime.now()
    startup_scripts = Path(__file__).parent / "startup_scripts" / now.strftime("%Y%m%d")
    startup_scripts.mkdir(parents=True, exist_ok=True)
    startup_file = startup_scripts / f"slurm_submit_{now.strftime('%Y%m%d_%H%M%S')}.sh"
    with open(startup_file, "w") as f:
        f.write(output)

    if dry_run:
        print(f"[DRY RUN] Script saved to: {startup_file}")
        return None

    if slurm_submit_type.lower() == "api":
        return _submit_slurmrestd(
            slurmrestd_urls=slurmrestd_urls,
            slurm_jwt_token=slurm_jwt_token,
            script=output,
            slurm_logdir=slurm_logdir,
            job_logdir=job_logdir,
            job_desc=job_desc,
            n_workers=n_workers,
            gpu_num=gpu_num,
            processes_per_worker=processes_per_worker,
            max_restarts=max_restarts,
            nodelist=nodelist,
            partition=partition,
        )

    return _submit_sbatch(startup_file, job_logdir, slurm_logdir)
