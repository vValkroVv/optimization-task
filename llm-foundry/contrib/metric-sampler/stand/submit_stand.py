import argparse
import sys

from stand import job_options

REQUIRED_CHECKOUT_FILES = (
    "scripts/train/train.py",
    "contrib/job_submitter/job_submitter/jobs/llmf/train.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submits a training job using the selected llm-foundry checkout.")
    parser.add_argument("config", help="config file in the configs directory of the stand root")
    parser.add_argument("n_workers", nargs="?", type=int, default=1)
    parser.add_argument("exp_name", nargs="?", default="metric-sampler-stand")
    job_options.add_arguments(parser)
    arguments = parser.parse_args()
    if arguments.n_workers < 1:
        parser.error("n_workers must be positive")
    for path in REQUIRED_CHECKOUT_FILES:
        if not (arguments.llmf_root / path).is_file():
            parser.error(f"selected checkout is missing {arguments.llmf_root / path}")

    sys.path.insert(0, str(arguments.llmf_root))
    from contrib.job_submitter.job_submitter.jobs.llmf.train import TrainJobLLMF

    TrainJobLLMF(
        exp_name=arguments.exp_name,
        path_to_config=str(arguments.stand_root / "configs" / arguments.config),
        n_workers=arguments.n_workers,
        **job_options.submission_options(arguments),
    ).submit()


if __name__ == "__main__":
    main()
