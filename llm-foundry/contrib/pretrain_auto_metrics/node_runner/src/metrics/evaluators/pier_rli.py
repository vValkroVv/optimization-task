from .harbor_rli import HarborRliEvaluator


class PierRliEvaluator(HarborRliEvaluator):
    NAME = "pier_rli"
    VERSION_FILE = "/home/user/pier_rli/pier_rli_version.txt"
    ENV_ROOT = "/home/user/pier_rli_env"
    REWARD_KEY = "reward"
    SHOOTER_NAMES = {}
