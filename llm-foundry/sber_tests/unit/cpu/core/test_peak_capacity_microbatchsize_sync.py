import pytest
from omegaconf import OmegaConf as om

from scripts.train.src.config_loader import sync_peak_capacity_microbatchsize


@pytest.mark.unit
@pytest.mark.fast
def test_sync_peak_capacity_microbatchsize_from_device_train_microbatch_size():
    cfg = om.create(
        {
            "device_train_microbatch_size": 4,
            "model": {
                "config_overrides": {
                    "dispatcher_balancing_strategy": "peak_capacity_factor",
                    "expert_capacity_factor": 1.2,
                }
            },
        }
    )

    sync_peak_capacity_microbatchsize(cfg)

    assert cfg.model.config_overrides.peak_capacity_microbatchsize == 4


@pytest.mark.unit
@pytest.mark.fast
def test_sync_peak_capacity_microbatchsize_preserves_explicit_override():
    cfg = om.create(
        {
            "device_train_microbatch_size": 4,
            "model": {
                "config_overrides": {
                    "dispatcher_balancing_strategy": "peak_capacity_factor",
                    "expert_capacity_factor": 1.2,
                    "peak_capacity_microbatchsize": 7,
                }
            },
        }
    )

    sync_peak_capacity_microbatchsize(cfg)

    assert cfg.model.config_overrides.peak_capacity_microbatchsize == 7


@pytest.mark.unit
@pytest.mark.fast
def test_sync_peak_capacity_microbatchsize_uses_student_model_overrides():
    cfg = om.create(
        {
            "device_train_microbatch_size": 3,
            "model": {
                "student_model": {
                    "config_overrides": {
                        "dispatcher_balancing_strategy": "peak_capacity_factor",
                        "expert_capacity_factor": 1.2,
                    }
                }
            },
        }
    )

    sync_peak_capacity_microbatchsize(cfg)

    assert cfg.model.student_model.config_overrides.peak_capacity_microbatchsize == 3


@pytest.mark.unit
@pytest.mark.fast
def test_sync_peak_capacity_microbatchsize_requires_integer_device_microbatch():
    cfg = om.create(
        {
            "device_train_microbatch_size": "auto",
            "model": {
                "config_overrides": {
                    "dispatcher_balancing_strategy": "peak_capacity_factor",
                    "expert_capacity_factor": 1.2,
                }
            },
        }
    )

    with pytest.raises(ValueError, match="peak_capacity_microbatchsize"):
        sync_peak_capacity_microbatchsize(cfg)
