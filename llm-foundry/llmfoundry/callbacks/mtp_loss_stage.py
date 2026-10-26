import copy
import logging

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger

from llmfoundry.callbacks.topk_accuracy_monitor import find_causal_lm

__all__ = ['MtpLossStage']

log = logging.getLogger(__name__)


class MtpLossStage(Callback):
    """Switches the MTP draft-loss weights once, mid-run:

        callbacks:
          mtp_loss_stage:
            switch_at: 500ba
            weights:
              lk_loss_weight: 0.0
              e2e_tv_loss_weight: 1.0

    Every entry is applied in the same step, not ramped (F-18). Weight 0 is not
    "off": such a kind is still computed and logged, which is what makes the
    switch a state change rather than a rebuild. Item 4 has what each of
    ``fit_start``'s guards is for.
    """

    def __init__(self, switch_at: str, weights: dict):
        if not weights:
            raise ValueError("MtpLossStage needs a non-empty `weights` mapping")
        self.switch_at = Time.from_timestring(switch_at)
        if self.switch_at.unit not in (TimeUnit.BATCH, TimeUnit.EPOCH, TimeUnit.TOKEN):
            raise ValueError(
                f"switch_at must be in batches, epochs or tokens, got {switch_at}"
            )
        self.weights = dict(weights)
        self._inner = None
        self._switched = False

    def fit_start(self, state: State, logger: Logger):
        inner = find_causal_lm(state)
        if inner is None or not getattr(inner, 'use_mtp', False):
            raise ValueError(
                "MtpLossStage is configured but the model has no MTP head"
            )
        kinds = set(getattr(inner, 'draft_loss_kinds', []))
        for name in self.weights:
            if not name.endswith("_loss_weight"):
                raise ValueError(
                    f"MtpLossStage weights must name a `<kind>_loss_weight`, got {name}"
                )
            if not hasattr(inner, name):
                raise ValueError(f"MtpLossStage: the model has no attribute {name}")
            kind = name[: -len("_loss_weight")]
            if kind != "mtp_ce" and kind not in kinds:
                raise ValueError(
                    f"MtpLossStage cannot stage in `{kind}`: it is not in "
                    f"draft_loss_kinds {sorted(kinds)}, which is frozen at "
                    f"construction. Set use_{kind}_loss=true (weight 0) so it is "
                    "computed from step 0."
                )
        self._validate_staged_config(inner.config)
        self._inner = inner
        current = {name: getattr(inner, name) for name in self.weights}
        log.info(f"MtpLossStage: at {self.switch_at} set {self.weights} (now {current})")
        self._maybe_switch(state)

    def _validate_staged_config(self, config):
        """Reject at `fit_start` a stage-2 state the config layer forbids, instead of
        at the first post-switch checkpoint reload.
        """
        staged = copy.deepcopy(config)
        for name, value in self.weights.items():
            setattr(staged, name, value)
        try:
            staged.validate_mtp_config()
        except ValueError as exc:
            raise ValueError(
                f"MtpLossStage: applying {self.weights} would leave the config in "
                f"a state the model rejects -- {exc}"
            ) from exc

    def batch_start(self, state: State, logger: Logger):
        self._maybe_switch(state)

    def _maybe_switch(self, state: State):
        if self._switched:
            return
        if state.timestamp.get(self.switch_at.unit) < self.switch_at:
            return
        for name, value in self.weights.items():
            setattr(self._inner, name, value)
            setattr(self._inner.config, name, value)
        self._switched = True
        log.info(
            f"MtpLossStage: switched to stage 2 at {state.timestamp.batch} -- "
            f"{self.weights}"
        )
