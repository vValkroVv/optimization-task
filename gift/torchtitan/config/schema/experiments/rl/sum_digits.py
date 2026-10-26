from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.experiments.rl.env import EnvConfig


@config(type="sum_digits")
class SumDigitsEnvConfig(EnvConfig):
    _owner_spec: ClassVar[str] = "torchtitan.experiments.rl.sum_digits:SumDigitsEnv"

    correctness_reward: float = 1.0
    """Reward for a response containing ``[ANSWER] <target>``."""

    format_reward: float = 0.3
    """Reward bonus for any ``[ANSWER] <number>`` tag in the response."""

    seed: int = 42
    """Seed mixed with ``(step, group_idx)`` to deterministically generate problems."""
