"""The two-stage cold-start -> end-to-end switch.

AngelSpec's two stages are the two branches of one `if/else` (angelspec.md).
Translated into our additive weights that is `lk_loss_weight` 1 -> 0 and
`e2e_tv_loss_weight` 0 -> 1 **in the same step**, not two independent ramps.

Pinned here, one class each: that the switch reaches the attribute the loss
reads and the config a checkpoint serializes, that a weight flipped on a kind
outside `draft_loss_kinds` raises rather than doing nothing, that the staged
config still validates, and that resuming past the switch point lands in stage 2.
Item 4 has what each of those costs when it is missed.

Ported from samoylovlesha/angelspec_eagle.
Reference: AngelSpec @ d3412be, `angelspec/training/mtp_trainer.py:521`
(`_backward` -- the if/else that makes the two stages mutually exclusive).
"""

import pytest
from torch import nn

from llmfoundry.callbacks.mtp_loss_stage import MtpLossStage


class _FakeConfig:
    def __init__(self):
        self.lk_loss_weight = 1.0
        self.e2e_tv_loss_weight = 0.0

    def validate_mtp_config(self):
        pass


class _FakeInner(nn.Module):
    """Only what the callback reads: it walks `state.model.model` to find this."""

    def __init__(self, draft_loss_kinds=("lk", "tv", "e2e_tv")):
        super().__init__()
        self.config = _FakeConfig()
        self.lm_head = nn.Linear(2, 2, bias=False)
        self.use_mtp = True
        self.draft_loss_kinds = list(draft_loss_kinds)
        self.lk_loss_weight = 1.0
        self.e2e_tv_loss_weight = 0.0


class _FakeModel(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.model = inner


class _FakeTimestamp:
    def __init__(self, batch):
        self.batch = batch

    def get(self, unit):
        from composer.core import Time

        return Time(self.batch, unit)


class _FakeState:
    def __init__(self, inner, batch=0):
        self.model = _FakeModel(inner)
        self.timestamp = _FakeTimestamp(batch)


STAGE2 = {"lk_loss_weight": 0.0, "e2e_tv_loss_weight": 1.0}


def _callback(switch_at="10ba", weights=None):
    return MtpLossStage(switch_at=switch_at, weights=dict(weights or STAGE2))


@pytest.mark.unit
@pytest.mark.fast
class TestMtpLossStage:
    def test_before_the_switch_nothing_moves(self):
        inner = _FakeInner()
        callback = _callback()
        state = _FakeState(inner, batch=0)
        callback.fit_start(state, None)
        for batch in (0, 5, 9):
            callback.batch_start(_FakeState(inner, batch=batch), None)
            assert inner.lk_loss_weight == 1.0
            assert inner.e2e_tv_loss_weight == 0.0

    def test_switch_applies_every_weight_in_the_same_step(self):
        """Both halves at once -- a staggered switch is a regime nobody ran."""
        inner = _FakeInner()
        callback = _callback()
        callback.fit_start(_FakeState(inner, batch=0), None)
        callback.batch_start(_FakeState(inner, batch=10), None)
        assert inner.lk_loss_weight == 0.0
        assert inner.e2e_tv_loss_weight == 1.0

    def test_the_attribute_is_set_not_only_the_config(self):
        """The loss reads the attribute; the config copy is for the checkpoint."""
        inner = _FakeInner()
        callback = _callback()
        callback.fit_start(_FakeState(inner, batch=0), None)
        callback.batch_start(_FakeState(inner, batch=10), None)
        assert inner.e2e_tv_loss_weight == 1.0
        assert inner.config.e2e_tv_loss_weight == 1.0

    def test_it_fires_once(self):
        inner = _FakeInner()
        callback = _callback()
        callback.fit_start(_FakeState(inner, batch=0), None)
        callback.batch_start(_FakeState(inner, batch=10), None)
        # A later hand edit must survive: the callback is a one-shot switch, not
        # a per-batch clamp.
        inner.e2e_tv_loss_weight = 0.25
        callback.batch_start(_FakeState(inner, batch=11), None)
        assert inner.e2e_tv_loss_weight == 0.25

    def test_resume_past_the_switch_lands_in_stage_2(self):
        """`fit_start` at batch 40 of a 10ba switch must not replay stage 1."""
        inner = _FakeInner()
        callback = _callback()
        callback.fit_start(_FakeState(inner, batch=40), None)
        assert inner.lk_loss_weight == 0.0
        assert inner.e2e_tv_loss_weight == 1.0

    def test_a_kind_outside_draft_loss_kinds_raises(self):
        """The trap: `draft_loss_kinds` is frozen, so the weight would do nothing."""
        inner = _FakeInner(draft_loss_kinds=("lk", "tv"))  # no e2e_tv
        callback = _callback()
        with pytest.raises(ValueError, match="cannot stage in `e2e_tv`"):
            callback.fit_start(_FakeState(inner, batch=0), None)

    def test_an_unknown_attribute_raises(self):
        inner = _FakeInner()
        callback = _callback(weights={"nonsense_loss_weight": 1.0})
        with pytest.raises(ValueError, match="no attribute nonsense_loss_weight"):
            callback.fit_start(_FakeState(inner, batch=0), None)

    def test_a_non_weight_key_raises(self):
        inner = _FakeInner()
        callback = _callback(weights={"mtp_predictor_num": 4})
        with pytest.raises(ValueError, match="must name a `<kind>_loss_weight`"):
            callback.fit_start(_FakeState(inner, batch=0), None)

    def test_a_model_without_mtp_raises(self):
        inner = _FakeInner()
        inner.use_mtp = False
        with pytest.raises(ValueError, match="no MTP head"):
            _callback().fit_start(_FakeState(inner, batch=0), None)

    def test_empty_weights_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="non-empty"):
            MtpLossStage(switch_at="10ba", weights={})

    def test_mtp_ce_weight_may_be_staged_without_being_a_draft_kind(self):
        """`mtp_ce_loss_weight` is always computed; it is not a draft kind."""
        inner = _FakeInner()
        inner.mtp_ce_loss_weight = 1.0
        inner.config.mtp_ce_loss_weight = 1.0
        callback = _callback(weights={"mtp_ce_loss_weight": 0.1})
        callback.fit_start(_FakeState(inner, batch=0), None)
        callback.batch_start(_FakeState(inner, batch=10), None)
        assert inner.mtp_ce_loss_weight == 0.1

    def test_the_registry_builds_it(self):
        from llmfoundry.utils.builders import build_callback

        built = build_callback(
            "mtp_loss_stage", {"switch_at": "500ba", "weights": dict(STAGE2)}
        )
        assert isinstance(built, MtpLossStage)
        assert built.weights == STAGE2
