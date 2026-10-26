import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent.parent))
from contrib.job_submitter.job_submitter.jobs.infinite import InfiniteJob

if __name__ == "__main__":
    InfiniteJob(
        user_login="default",
        exp_name="default",
        n_workers=1,
        max_retry=None,
        job_logdir=None,
        slurm_logdir=None,
        extra_env={},
        queue_name=None, # qos or queue name
        priority_class=None # priority class, partition depends on it
    ).submit()
