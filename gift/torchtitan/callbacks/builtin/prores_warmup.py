from typing import TypeAlias

from torchtitan.callbacks.base import MutableState, Mutation, Mutator
from torchtitan.callbacks.builtin.autometrics import Autometrics
from torchtitan.callbacks.sources.base import canonical_fqn
from torchtitan.config import Configurable
from torchtitan.config.schema.callbacks.builtin.prores_warmup import (
    ProResWarmupConfig,
)
from torchtitan.models.common.residual_scale import ResidualScale
from torchtitan.tools.logging import logger


class ProResWarmup(Configurable, Mutator):
    """Feeds the step to every ``ResidualScale``; the schedule is on each module.

    Stateless: ``alpha`` is recomputed from ``trainer.step`` on resume. Each
    pipeline stage moves its own modules.

    Logged metrics when ``emit_metrics`` is enabled:
        - ``residual_scale/<module fqn>``: that module's ``alpha``
    """

    mutates = frozenset({Mutation.residual_scale})

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = ProResWarmupConfig

    def __init__(self, config: Config) -> None:
        self.config = config
        self._modules: list[tuple[str, ResidualScale]] = []
        self._autometrics: Autometrics | None = None

    def fit_start(self, state: MutableState) -> None:
        self._modules = [
            (canonical_fqn(fqn), module)
            for model_part in state.trainer.model_parts
            for fqn, module in model_part.named_modules()
            if isinstance(module, ResidualScale)
        ]
        if not self._modules:
            logger.warning(
                "ProResWarmup: no ResidualScale module on this rank; "
                "the callback does nothing here."
            )
        self._autometrics = state.trainer.callbacks.find(Autometrics)
        # The next batch is step + 1; see batch_start.
        self._set_step(state.step)

    def batch_start(self, state: MutableState) -> None:
        # trainer.step is 1-based, the schedule 0-based.
        alphas = self._set_step(state.step - 1)
        if self.config.emit_metrics and state.step % self.config.log_interval == 0:
            for (fqn, _), alpha in zip(self._modules, alphas):
                state.record(f"residual_scale/{fqn}", alpha)

    def after_checkpoint_save(self, state: MutableState) -> None:
        # The trainer's HF saves are checked before the run
        # (``validate_hf_export``); Autometrics converts DCP saves on its own
        # interval, the only HF path on gfsdp2.
        if (
            self._autometrics is None
            or state.step % self._autometrics.config.interval != 0
        ):
            return
        step = state.step - 1
        unsaturated = [
            fqn for fqn, module in self._modules if module.alpha_at(step) < 1.0
        ]
        if unsaturated:
            logger.warning(
                f"Checkpoint at step {state.step} goes to HF: {len(unsaturated)} "
                "ResidualScale modules still have alpha < 1 (first: "
                f"{unsaturated[0]}). The HF model has no residual scale and "
                "computes as if alpha == 1."
            )

    def _set_step(self, step: int) -> list[float]:
        return [module.set_step(step) for _, module in self._modules]
