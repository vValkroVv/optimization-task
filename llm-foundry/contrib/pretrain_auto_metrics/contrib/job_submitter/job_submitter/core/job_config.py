from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(slots=True)
class JobTemplate:
    submission_type: str = "cloud"
    user_login: str = "default"
    exp_name: str = "default"
    job_desc: str = "default"
    base_image: Optional[str] = None
    script: str = ""
    n_workers: int = 1
    processes_per_worker: int = 1
    script_type: str = "binary"

    env_variables: Dict[str, str] = field(default_factory=dict)
    region: str = "SR008"
    max_retry: Optional[int] = None
    gpu_num: int = 8
    cpu_only: bool = False
    gpu_type: Optional[str] = None
    instance_type: Optional[str] = None
    priority_class: Optional[str] = "medium"
    allocation_name: Optional[str] = None
    internet: bool = True  # cloud only
    queue_name: str = "shared"

    slurm_jwt_token: Optional[str] = field(default=None, repr=False)  # slurm only
    slurmrestd_urls: list[str] = field(default_factory=list)  # slurm only
    slurm_submit_type: str = "sbatch"  # slurm only
    job_logdir: Optional[str] = None  # slurm only
    slurm_logdir: Optional[str] = None  # slurm only
    partition: str = "middle"  # slurm only
    cpus_per_task: int = 192  # slurm only
    memory: Optional[str] = None  # slurm only
    time_limit: Optional[str] = None  # slurm only
    requeue: bool = False  # slurm only
    comment: Optional[str] = None  # slurm only
    turn_off_first_srun: bool = False  # slurm only
    extra_mounts: Optional[str] = None  # slurm only
    qos: Optional[str] = None  # slurm only
    account: Optional[str] = None  # slurm only
    nodelist: Optional[str] = None  # slurm only
    nice: Optional[int] = None  # slurm only
    flags: Dict | None = field(default_factory=dict)
    exclude_nodes: Optional[str] = None  # slurm only
