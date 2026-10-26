"""Base class of the RL policy losses (``policy_trainer.loss``).

Not a ``torchtitan.components.loss.BaseLoss``: the call signature (logprobs,
advantages, masks) differs from the LM losses.
"""

from torchtitan.config import Configurable
from typing import TypeAlias

from torchtitan.config.schema.experiments.rl.loss import RLLossConfig


class RLLoss(Configurable):
    Config: TypeAlias = RLLossConfig
