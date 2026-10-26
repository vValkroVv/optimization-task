from contrib.job_submitter.job_submitter.jobs.pam.autometrics import AutometricsJobPAM


CHECKPOINT_DIR = ""  # /home/user/deepseek/DeepSeek-V3.1

# Can remain empty if config is in <checkpoint directory>/artifacts/llm_foundry_config.yaml is valid for autometrics
# ./configs/700b.yaml - example
PATH_TO_CONFIG = ""

if __name__ == "__main__":
    AutometricsJobPAM(
        user_login="default",
        exp_name=f"Autometrics run for checkpoint {CHECKPOINT_DIR}",
        checkpoint_dir=CHECKPOINT_DIR,
        path_to_config=PATH_TO_CONFIG,
        queue_name="shared",
    ).submit()
