import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent.parent))
from contrib.job_submitter.job_submitter.jobs.llmf.autometrics import AutometricsJobLLMF

CHECKPOINT_DIR = ""  # /home/user/deepseek/DeepSeek-V3.1

# Can remain empty if config is in <checkpoint directory>/artifacts/llm_foundry_config.yaml is valid for autometrics
# <path to llm-foundry>/contrib/pretrain_auto_metrics/node_runner/configs/700b.yaml - example
PATH_TO_CONFIG = ""

if __name__ == "__main__":
    AutometricsJobLLMF(
        user_login="default",
        exp_name=f"Autometrics run for checkpoint {CHECKPOINT_DIR}",
        checkpoint_dir=CHECKPOINT_DIR,
        path_to_config=PATH_TO_CONFIG,
        n_workers=1,
        enable_auto_track=True,  # 🚀 Want this run to show up in Mattermost automatically? ✨
        # ✨ Set Auto-tracking to True and you will get notifications in mattermost! 💪💪💪
        # if you submit from slurm and doesn't get notifications check yourself in mapping
        # Mapping of username nicknames is here in gitlab
        # papayallms/papayachat_tracker_bot/-/blob/main/configs/username_mapping.json
        # don't forget to change papaya to ... not papaya. 🥴🍑
        # and don't forget to set your correct mattermost account or papaya slurm name
        max_retry=None,
        job_logdir=None,
        slurm_logdir=None,
        extra_env={},
        queue_name=None, # qos or queue name
        priority_class=None # priority class, partition depends on it
    ).submit()
