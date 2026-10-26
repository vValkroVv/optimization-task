from job_templates.region_configs import (
    SLURM_REGIONS,
    CLOUD_REGIONS,
    DEFAULT_REGION_ENVS, 
    DEFAULT_SLURM_REGION_PARAMS,
    get_instance_type
)
from job_templates.slurm_client_lib import submit_slurm_job

__all__ = [
    "SLURM_REGIONS",
    "CLOUD_REGIONS",
    "DEFAULT_REGION_ENVS",
    "DEFAULT_SLURM_REGION_PARAMS",
    "get_instance_type",
    "submit_slurm_job",
]