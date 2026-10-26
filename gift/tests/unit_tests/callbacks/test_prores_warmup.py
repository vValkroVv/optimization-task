"""ProResWarmup: feeds the step to every ResidualScale (CPU)."""

import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch.nn as nn

from torchtitan.callbacks import ProResWarmup
from torchtitan.callbacks.base import MutableState
from torchtitan.models.common.residual_scale import (
    ResidualScale,
    ResidualScaleSchedule,
)

from tests.unit_tests.callbacks.trainer_stub import make_trainer

_TRAINING_STEPS = 100


def _scale(
    warmup_steps: float,
    schedule: ResidualScaleSchedule = ResidualScaleSchedule.linear,
) -> ResidualScale:
    return ResidualScale.Config(
        warmup_steps=warmup_steps, start_value=0.0, schedule=schedule
    ).build()


def _part(
    *layers: tuple[int, float],
    schedule: ResidualScaleSchedule = ResidualScaleSchedule.linear,
) -> nn.Module:
    """A model part with one ResidualScale per layer, keyed like a decoder."""
    part = nn.Module()
    part.layers = nn.ModuleDict(
        {
            str(i): nn.ModuleDict({"attn_residual_scale": _scale(w, schedule)})
            for i, w in layers
        }
    )
    return part


def _trainer(
    model_parts: list[nn.Module],
    *,
    step: int,
    autometrics_interval: int | None = None,
) -> SimpleNamespace:
    autometrics = (
        None
        if autometrics_interval is None
        else SimpleNamespace(config=SimpleNamespace(interval=autometrics_interval))
    )
    return make_trainer(
        step=step,
        model_parts=model_parts,
        callbacks=SimpleNamespace(find=lambda _type: autometrics),
        config=SimpleNamespace(training=SimpleNamespace(steps=_TRAINING_STEPS)),
        checkpointer=SimpleNamespace(last_saved_path=None),
    )


def _alphas(model_parts: list[nn.Module]) -> dict[str, Any]:
    return {
        fqn: module.alpha.item()
        for part in model_parts
        for fqn, module in part.named_modules()
        if isinstance(module, ResidualScale)
    }


def test_first_batch_uses_step_zero() -> None:
    parts = [_part((0, 10.0), (1, 20.0))]
    trainer = _trainer(parts, step=0)
    callback = ProResWarmup(ProResWarmup.Config())
    callback.fit_start(MutableState(trainer=trainer))
    trainer.step = 1
    callback.batch_start(MutableState(trainer=trainer))
    assert set(_alphas(parts).values()) == {0.0}

    trainer.step = 6  # schedule step 5
    callback.batch_start(MutableState(trainer=trainer))
    assert _alphas(parts) == {
        "layers.0.attn_residual_scale": pytest.approx(0.5),
        "layers.1.attn_residual_scale": pytest.approx(0.25),
    }


def test_every_pipeline_stage_moves_its_own_layers() -> None:
    # Stages keep global layer indices in their FQNs.
    parts = [_part((0, 10.0)), _part((1, 20.0))]
    trainer = _trainer(parts, step=0)
    callback = ProResWarmup(ProResWarmup.Config())
    callback.fit_start(MutableState(trainer=trainer))
    trainer.step = 11
    callback.batch_start(MutableState(trainer=trainer))
    assert _alphas(parts[:1]) == {"layers.0.attn_residual_scale": 1.0}
    assert _alphas(parts[1:]) == {"layers.1.attn_residual_scale": pytest.approx(0.5)}


def test_resume_recomputes_alpha_from_step() -> None:
    # Continuous run through step 8.
    continuous = [_part((0, 10.0), (1, 20.0))]
    trainer = _trainer(continuous, step=0)
    callback = ProResWarmup(ProResWarmup.Config())
    callback.fit_start(MutableState(trainer=trainer))
    for step in range(1, 9):
        trainer.step = step
        callback.batch_start(MutableState(trainer=trainer))

    # Fresh modules and callback, resumed at step 7.
    resumed = [_part((0, 10.0), (1, 20.0))]
    trainer = _trainer(resumed, step=7)
    callback = ProResWarmup(ProResWarmup.Config())
    callback.fit_start(MutableState(trainer=trainer))
    assert _alphas(resumed) == _alphas(continuous)
    trainer.step = 8
    callback.batch_start(MutableState(trainer=trainer))
    assert _alphas(resumed) == _alphas(continuous)


def test_metrics_only_when_enabled_and_on_interval() -> None:
    parts = [_part((0, 10.0))]
    trainer = _trainer(parts, step=0)
    callback = ProResWarmup(ProResWarmup.Config(emit_metrics=True, log_interval=5))
    callback.fit_start(MutableState(trainer=trainer))

    trainer.step = 4
    state = MutableState(trainer=trainer)
    callback.batch_start(state)
    assert state._records == []

    trainer.step = 5
    state = MutableState(trainer=trainer)
    callback.batch_start(state)
    [record] = state._records
    assert record.name == "residual_scale/layers.0.attn_residual_scale"
    assert record.value == pytest.approx(0.4)


def test_metric_names_drop_wrapper_segments() -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    part = _part((0, 10.0))
    layers = part.get_submodule("layers")
    assert isinstance(layers, nn.ModuleDict)
    layers["0"] = checkpoint_wrapper(layers["0"])
    trainer = _trainer([part], step=0)
    callback = ProResWarmup(ProResWarmup.Config(emit_metrics=True, log_interval=1))
    callback.fit_start(MutableState(trainer=trainer))
    trainer.step = 1
    state = MutableState(trainer=trainer)
    callback.batch_start(state)
    [record] = state._records
    assert record.name == "residual_scale/layers.0.attn_residual_scale"


def test_no_metrics_by_default() -> None:
    parts = [_part((0, 10.0))]
    trainer = _trainer(parts, step=0)
    callback = ProResWarmup(ProResWarmup.Config())
    callback.fit_start(MutableState(trainer=trainer))
    trainer.step = 10
    state = MutableState(trainer=trainer)
    callback.batch_start(state)
    assert state._records == []


def test_rejects_non_positive_log_interval() -> None:
    with pytest.raises(ValueError):
        ProResWarmup.Config(log_interval=0)


def test_warns_without_modules(caplog: pytest.LogCaptureFixture) -> None:
    trainer = _trainer([nn.Linear(2, 2)], step=0)
    with caplog.at_level(logging.WARNING):
        ProResWarmup(ProResWarmup.Config()).fit_start(MutableState(trainer=trainer))
    assert any("no ResidualScale" in m for m in caplog.messages)


@pytest.mark.parametrize(
    ("step", "interval", "schedule", "warns"),
    [
        (5, 5, ResidualScaleSchedule.linear, True),  # submitted, mid-warmup
        (6, 5, ResidualScaleSchedule.linear, False),  # not a submit step
        (50, 5, ResidualScaleSchedule.linear, False),  # submitted, saturated
        (50, 5, ResidualScaleSchedule.cosine, False),  # cosine saturates to exactly 1
    ],
)
def test_autometrics_conversion_warning(
    caplog: pytest.LogCaptureFixture,
    step: int,
    interval: int,
    schedule: ResidualScaleSchedule,
    warns: bool,
) -> None:
    # Trainer HF saves are rejected before the run; Autometrics converts DCP.
    parts = [_part((0, 10.0), schedule=schedule)]
    trainer = _trainer(parts, step=step, autometrics_interval=interval)
    callback = ProResWarmup(ProResWarmup.Config())
    callback.fit_start(MutableState(trainer=trainer))
    with caplog.at_level(logging.WARNING):
        callback.after_checkpoint_save(MutableState(trainer=trainer))
    assert any("alpha < 1" in m for m in caplog.messages) == warns
