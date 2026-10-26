import getpass
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


from job_submitter.core.job_templates.region_configs import (
    DEFAULT_SLURM_REGION_TEMPLATE_PARAMS,
)
from job_submitter.core.slurm_rest import SlurmRestClient

_PYTORCH2_SCRIPT = """\
python -m torch.distributed.launch \\
    --nproc-per-node=\\$JS_GPUS_PER_NODE \\
    --nnodes=\\$NNODES \\
    --node-rank=\\$SLURM_PROCID \\
    --master-addr=\\$SLURM_LAUNCH_NODE_IPADDR \\
    --master-port=12345 \\
    --use-env \\
    {script}
"""


def _memory_to_mib(memory: Optional[str]) -> Optional[int]:
    if memory is None:
        return None
    match = re.fullmatch(r"(\d+)([MGT]?)", memory.strip(), re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unsupported Slurm memory value: {memory}")
    value = int(match.group(1))
    multiplier = {"": 1, "M": 1, "G": 1024, "T": 1024**2}[match.group(2).upper()]
    return value * multiplier


def _time_limit_to_minutes(time_limit: Optional[str]) -> Optional[int]:
    if time_limit is None:
        return None
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", time_limit.strip())
    if match is None:
        raise ValueError(f"Unsupported Slurm time limit: {time_limit}")
    days, hours, minutes, seconds = (int(value or 0) for value in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Unsupported Slurm time limit: {time_limit}")
    total_seconds = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    return (total_seconds + 59) // 60


def _submit_sbatch(script_path, job_logdir, slurm_logdir):
    try:
        print(f"Submitting sbatch with script: {script_path}")
        env = os.environ.copy()
        env["SLURM_UMASK"] = "0002"
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

        print("Submission output:", result.stdout)
        print("Submission error (if any):", result.stderr)

        job_id = result.stdout.split(" ")[-1].strip()
        if not result.stderr:
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
        else:
            print(
                f"[WARNING] sbatch returned stderr; job id parsed from stdout: {job_id}"
            )
        return job_id

    except subprocess.CalledProcessError as e:
        print("Error while submitting job:")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error message:", e.stderr)
        raise


def _rest_environment_entries() -> list[str]:
    """REST jobs get an explicit environment (unlike sbatch, which inherits the
    submitter's). Pass HOME/PATH through so tools that resolve ~/.config or
    exec helpers (e.g. rclone -> getent) behave like under sbatch."""
    entries = [f"USER={getpass.getuser()}"]
    for name in ("HOME", "PATH"):
        value = os.environ.get(name)
        if value:
            entries.append(f"{name}={value}")
    return entries


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
    gpu_type: Optional[str] = None,
    nodelist: Optional[str] = None,
    partition: str = "middle",
    qos: Optional[str] = None,
    account: Optional[str] = None,
    exclude_nodes: Optional[str] = None,
    nice: Optional[int] = None,
    cpu_only: bool = False,
    cpus_per_task: Optional[int] = None,
    memory: Optional[str] = None,
    time_limit: Optional[str] = None,
    requeue: bool = False,
    comment: Optional[str] = None,
):
    standard_output = f"{slurm_logdir}/%j/logs.log"
    standard_error = f"{slurm_logdir}/%j/logs.err"
    if cpu_only:
        standard_output = f"{slurm_logdir}/slurm-%j.out"
        standard_error = f"{slurm_logdir}/slurm-%j.err"

    job = {
        "name": job_desc,
        "standard_output": standard_output,
        "standard_error": standard_error,
        "tasks_per_node": processes_per_worker,
        "current_working_directory": str(Path(__file__).parent),
        "environment": {"USER": getpass.getuser()}
        if gpu_type is None and not cpu_only
        else _rest_environment_entries(),
        "partition": partition,
        "nodes": n_workers,
    }
    if cpu_only:
        if cpus_per_task is not None:
            job["cpus_per_task"] = cpus_per_task
        memory_mib = _memory_to_mib(memory)
        if memory_mib is not None:
            job["memory_per_node"] = memory_mib
    else:
        job["tres_per_node"] = (
            f"gres/gpu:{gpu_num}"
            if gpu_type is None
            else f"gres/gpu:{gpu_type}:{gpu_num}"
        )
        job["exclusive"] = True
    time_limit_minutes = _time_limit_to_minutes(time_limit)
    if time_limit_minutes is not None:
        job["time_limit"] = time_limit_minutes
    if requeue:
        job["requeue"] = True
    if comment:
        job["comment"] = comment

    payload = {
        "script": script,
        "job": job,
    }
    if qos is not None:
        payload["job"]["qos"] = qos
    if account:
        payload["job"]["account"] = account
    if max_restarts:
        payload["job"]["max_restarts"] = max_restarts
        payload["job"]["requeue"] = True
    if nodelist:
        payload["job"]["nodelist"] = nodelist
    if exclude_nodes:
        payload["job"]["excluded_nodes"] = exclude_nodes
    if nice:
        payload["job"]["nice"] = nice

    job_id = SlurmRestClient(
        urls=slurmrestd_urls,
        jwt_token=slurm_jwt_token,
    ).submit(payload)

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
    cpu_only: bool = False,
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
    extra_mounts: Optional[str] = None,
    cpus_per_task: Optional[int] = None,
    memory: Optional[str] = None,
    time_limit: Optional[str] = None,
    requeue: bool = False,
    comment: Optional[str] = None,
    turn_off_first_srun: bool = False,
    qos: Optional[str] = None,
    account: Optional[str] = None,
    gpu_type: Optional[str] = None,
    exclude_nodes=None,
    nice: Optional[int] = None,
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

    if extra_mounts:
        for mount in extra_mounts.split(","):
            mount = mount.strip()
            if not mount:
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
        "exclude_nodes": exclude_nodes,
        "nice": nice,
        "env_variables": env_variables or {},
        "partition": partition,
        "extra_mounts": extra_mounts,
        "cpus_per_task": cpus_per_task,
        "memory": memory,
        "time_limit": time_limit,
        "requeue": requeue,
        "comment": comment,
        "turn_off_first_srun": turn_off_first_srun,
        "qos": qos,
        "account": account,
        "gpu_type": gpu_type,
        **DEFAULT_SLURM_REGION_TEMPLATE_PARAMS.get(region, {}),
        **override_kwargs,
    }

    template_name = "cpu.sh.j2" if cpu_only else "binary.sh.j2"
    template_dir = Path(__file__).parent
    # Rendering is the only step that needs jinja2, and slim runtime images
    # (PAM) ship without it, so importing job_submitter must not require it.
    from jinja2 import Environment, FileSystemLoader

    environment = Environment(loader=FileSystemLoader(template_dir))
    template = environment.get_template(template_name)
    output = template.render(context)

    now = datetime.now()
    startup_scripts = Path(__file__).parent / "startup_scripts" / now.strftime("%Y%m%d")
    startup_scripts.mkdir(parents=True, exist_ok=True)
    startup_file = (
        startup_scripts / f"slurm_submit_{now.strftime('%Y%m%d_%H%M%S_%f')}.sh"
    )
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
            exclude_nodes=exclude_nodes,
            nice=nice,
            partition=partition,
            qos=qos,
            account=account,
            gpu_type=gpu_type,
            cpu_only=cpu_only,
            cpus_per_task=cpus_per_task,
            memory=memory,
            time_limit=time_limit,
            requeue=requeue,
            comment=comment,
        )

    return _submit_sbatch(startup_file, job_logdir, slurm_logdir)
