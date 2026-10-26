import functools
import math
from typing import ClassVar, Literal

from torchtitan.config.configurable import config, Configurable
from torchtitan.config.errors import ConfigFieldError
from torchtitan.tools.logging import logger


@config
class LRSchedulersContainerConfig(Configurable.Config):
    _owner_spec: ClassVar[str] = (
        "torchtitan.components.lr_scheduler:LRSchedulersContainer"
    )

    warmup_steps: int = 200
    """
    Steps for lr scheduler warmup, normally 1/5 of --training.steps
    """

    total_steps: int | None = None
    """Total steps of the LR schedule; None means ``training.steps``."""

    decay_ratio: float | None = None
    """
    Controls the proportion of the training steps allocated to the learning rate decay phase.
    If `None`, the learning rate will begin decaying immediately after the warmup period.
    Otherwise, the learning rate will remain stable after the warmup period and
    only start decaying during the last `decay_ratio` portion of the total training steps.
    This is known as the Warmup-Stable-Decay (WSD) schedule, as described in https://arxiv.org/abs/2404.06395.
    """

    decay_type: Literal["linear", "sqrt", "cosine"] = "linear"
    """
    Learning rate decay type to use during training:
    - 'linear': linearly decays learning rate from initial to final value
    - 'sqrt': decays learning rate following a 1 minus square root curve
    - 'cosine': smoothly decays learning rate following a cosine curve
    """

    min_lr_factor: float = 0.0
    """
    Min lr ratio for lr scheduler.
    If provided, the range of decay factor is scaled from 1 to `min_lr_factor`
    to ensure the learning rate does not drop below `optimizer.lr * lr_scheduler.min_lr_factor`.
    """

    def __post_init__(self) -> None:
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ConfigFieldError("warmup_steps", "must be a nonnegative integer")
        if self.total_steps is not None and (
            type(self.total_steps) is not int or self.total_steps <= 0
        ):
            raise ConfigFieldError("total_steps", "must be a positive integer")
        for name in ("min_lr_factor", "decay_ratio"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ConfigFieldError(name, "must be finite and in [0, 1]")

    # pyrefly: ignore [bad-override]
    def build(self, *, optimizers, training_steps):
        """Build a LRSchedulersContainer from this config.

        Args:
            optimizers: The corresponding OptimizersContainer.
            training_steps: The total number of training steps.

        Returns:
            A LRSchedulersContainer for the given optimizers.
        """
        total_steps = (
            self.total_steps if self.total_steps is not None else training_steps
        )

        warmup_steps = int(self.warmup_steps)

        if warmup_steps > total_steps:
            logger.warning(
                f"Warmup steps ({warmup_steps}) exceed total steps ({total_steps}). "
                f"Adjusting warmup steps to {total_steps}."
            )
            warmup_steps = total_steps

        if self.decay_ratio is not None:
            decay_steps = round(total_steps * self.decay_ratio)
            if warmup_steps + decay_steps > total_steps:
                logger.warning(
                    f"Warmup ({warmup_steps}) + decay ({decay_steps}) steps exceed "
                    f"total steps ({total_steps}). "
                    f"Adjusting decay steps to {total_steps - warmup_steps}."
                )
                decay_steps = total_steps - warmup_steps
        else:
            decay_steps = total_steps - warmup_steps
        # Add a virtual last step to prevent the learning rate from dropping to 0
        stable_steps = total_steps + 1 - warmup_steps - decay_steps
        lr_decay_type = self.decay_type
        min_lr_factor = self.min_lr_factor

        def linear_warmup_stable_decay(
            current_step: int,
            warmup_steps: int,
            stable_steps: int,
            decay_steps: int,
            lr_decay_type: str,
            min_lr_factor: float,
        ):
            """
            Computes linear warmup followed by stable learning rate for a while,
            then some type of decay.

            Per LambdaLR requirement, this is accomplished by returning
            a multiplicative factor `curr_adjustment` ranging from 1 to 0
            to adjust the learning rate to create the desired schedule.

            We offer three types of learning rate decay schedules:
            1. `linear`: decays linearly from 1 to 0 over the decay period.
            2. `sqrt`: decays as 1 minus the square root of the decay progress.
            3. `cosine`: follows a cosine curve, decaying according to the values of the half-period of the cosine function.

            If `min_lr_factor` is specified, the decay range is scaled from 1 to `min_lr_factor`
            to ensure the learning rate does not drop below this minimum value.
            """
            if current_step >= total_steps:
                return min_lr_factor if decay_steps else 1.0
            warmup_stable_steps = warmup_steps + stable_steps
            if current_step < warmup_steps:
                # linear warmup
                # 0-indexed step, hence + 1 adjustments
                current_step += 1
                assert warmup_steps != 0, (
                    "warmup_steps must not be zero to reach this branch"
                )
                curr_adjustment = float(current_step / warmup_steps)
            elif current_step < warmup_stable_steps:
                curr_adjustment = 1.0
            else:
                # 0-indexed step, hence + 1 adjustments
                current_step += 1
                assert decay_steps != 0, (
                    "decay_steps must not be zero to reach this branch"
                )
                progress = float(current_step - warmup_stable_steps) / decay_steps

                if lr_decay_type == "linear":
                    curr_adjustment = 1 - progress
                elif lr_decay_type == "sqrt":
                    curr_adjustment = 1 - math.sqrt(progress)
                elif lr_decay_type == "cosine":
                    curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
                else:
                    raise ValueError(f"Unknown lr_decay_type: {lr_decay_type}")
                curr_adjustment = min_lr_factor + (1 - min_lr_factor) * curr_adjustment
            return curr_adjustment

        lr_lambda = functools.partial(
            linear_warmup_stable_decay,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            lr_decay_type=lr_decay_type,
            min_lr_factor=min_lr_factor,
        )
        return self.owner()(optimizers, lr_lambda)
