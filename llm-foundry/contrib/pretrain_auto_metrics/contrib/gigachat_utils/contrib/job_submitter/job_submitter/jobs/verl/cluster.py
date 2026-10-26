from dataclasses import dataclass
from job_submitter.jobs.verl.base import RayBaseJobVERL


@dataclass(slots=True, kw_only=True)
class RayClusterJobVERL(RayBaseJobVERL):
    def _script(self) -> str:
        if self.submission_type == "cloud":
            return f"bash {self.verl_root}/sber_src/submit/setup_ray_cluster/cloud.sh"
        else:
            assert False, "Ray cluster is not supported for slurm yet"

    def _before_submit(self) -> None:
        pass
