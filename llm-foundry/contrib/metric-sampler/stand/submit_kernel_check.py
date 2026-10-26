import argparse
import sys

from stand import job_options


def main() -> None:
    parser = argparse.ArgumentParser(description="Submits a GPU check of stand/ as a one-node job.")
    job_options.add_arguments(parser)
    parser.add_argument("--base-image", required=True, help="container image the job runs in")
    parser.add_argument("--repo-root", type=job_options.resolved_path, required=True,
                        help="checkout of this repository that the job can read")
    parser.add_argument("check", nargs="?", default="check_activation_kernel.py",
                        help="script of the stand directory to run, for example check_token_loss_kernel.py")
    parser.add_argument("check_arguments", nargs=argparse.REMAINDER, help="arguments of the check")
    arguments = parser.parse_args()

    sys.path.insert(0, str(arguments.llmf_root))
    from contrib.job_submitter.job_submitter.jobs.any_script import AnyScriptJob

    repo = arguments.repo_root
    check_arguments = " ".join(arguments.check_arguments)
    AnyScriptJob(
        exp_name="metric-sampler-kernel-check",
        script=f"cd {repo} && PYTHONPATH={repo} python {repo}/stand/{arguments.check} {check_arguments}",
        base_image=arguments.base_image,
        n_workers=1,
        n_gpus=job_options.GPUS_PER_NODE,
        **job_options.submission_options(arguments),
    ).submit()


if __name__ == "__main__":
    main()
