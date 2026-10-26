from contrib.job_submitter.job_submitter.jobs.pam.autometrics_external_api import (
    AutometricsJobPAMExternalAPI,
)

PATH_TO_CONFIG = "./configs/ift.yaml"

if __name__ == "__main__":
    AutometricsJobPAMExternalAPI(
        user_login="default",
        exp_name=f"External API PAM run with config {PATH_TO_CONFIG}",
        path_to_config=PATH_TO_CONFIG,
        queue_name="shared",
        # output_path="./output", if needed
    ).submit()
