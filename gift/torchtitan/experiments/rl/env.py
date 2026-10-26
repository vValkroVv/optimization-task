"""Base class of the rollout environments (the ``env`` / ``validation_env``
sections); ``build(step=..., group_idx=...)`` yields one episode's environment.
"""

from torchtitan.config import Configurable
from typing import TypeAlias

from torchtitan.config.schema.experiments.rl.env import EnvConfig


class Env(Configurable):
    Config: TypeAlias = EnvConfig
