import argparse
import sys

from stand import job_options


def main() -> None:
    parser = argparse.ArgumentParser(description="Submits the shared-memory node probe as a one-node job.")
    job_options.add_arguments(parser)
    parser.add_argument("--base-image", required=True, help="container image the job runs in")
    arguments = parser.parse_args()

    sys.path.insert(0, str(arguments.llmf_root))
    from contrib.job_submitter.job_submitter.jobs.any_script import AnyScriptJob

    probe = arguments.stand_root / "shm_node_probe.py"
    AnyScriptJob(
        exp_name="metric-sampler-shm-probe",
        script=f"python -m torch.distributed.run --standalone --nproc-per-node {job_options.GPUS_PER_NODE} {probe}",
        base_image=arguments.base_image,
        n_workers=1,
        n_gpus=job_options.GPUS_PER_NODE,
        **job_options.submission_options(arguments),
    ).submit()


if __name__ == "__main__":
    main()
