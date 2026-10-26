import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent.parent))
from contrib.job_submitter.job_submitter.jobs.autometrics import AutometricsJob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AutometricsJob with specified checkpoint and config"
    )
    parser.add_argument(
        "--checkpoint-dir", required=True, help="Local path to checkpoint directory"
    )
    parser.add_argument(
        "--config-path", required=True, help="Local path to YAML config file"
    )
    parser.add_argument("--user", required=True, help="Username")
    parser.add_argument("--priority-class", default=None, help="Priority class")
    parser.add_argument("--queue-name", default="shared", help="Queue name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    retry_count = 3
    retry_index = 0
    success = False
    while retry_index < retry_count:
        submit_job_result = AutometricsJob(
            user_login=args.user,
            exp_name=f"Autometrics run for checkpoint {args.checkpoint_dir}",
            checkpoint_dir=args.checkpoint_dir,
            path_to_config=args.config_path,
            enable_auto_track=True, # 🚀 Want this run to show up in Mattermost automatically? ✨
            # ✨ Set Auto-tracking to True and you will get notifications in mattermost! 💪💪💪
            # if you submit from slurm and doesn't get notifications check yourself in mapping
            # Mapping of username nicknames is here in gitlab
            # papayallms/papayachat_tracker_bot/-/blob/main/configs/username_mapping.json
            # don't forget to change papaya to ... not papaya. 🥴🍑
            # and don't forget to set your correct mattermost account or papaya slurm name
            priority_class=args.priority_class,
            queue_name=args.queue_name,
        ).submit()

        if (
            isinstance(submit_job_result, str)
            and submit_job_result.startswith("Job ")
            and submit_job_result.endswith(" created.")
        ):
            print(f"🎉 Success! {submit_job_result}")
            success = True
            break

        print(f"🔁 Failed try №{retry_index + 1}. ")
        retry_index += 1
        time.sleep(10)

    if not success:
        raise Exception(submit_job_result)
