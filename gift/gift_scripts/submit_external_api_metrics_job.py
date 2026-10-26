import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "contrib" / "job_submitter"))

from job_submitter.jobs.gift.autometrics_external_api import (  # noqa: E402
    AutometricsJobGIFTExternalAPI,
)

if __name__ == "__main__":
    AutometricsJobGIFTExternalAPI(
        user_login="<user_login>",
        exp_name="<exp_name>",
        path_to_config="<path to external API PAM config>",
        queue_name="shared",
        gift_root_path=REPO_ROOT,
    ).submit()
