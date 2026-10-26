# Copyright 2024 LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for autometrics callback -> autometrics job kwarg forwarding."""

import pytest
from omegaconf import OmegaConf

pytest.importorskip("composer")

from llmfoundry.callbacks.autometrics_callback import Autometrics  # noqa: E402


def _config(**overrides):
    base = {"user": "u", "nnodes": 2, "eval_interval": 100}
    base.update(overrides)
    return OmegaConf.create(base)


def test_forwards_arbitrary_job_kwargs() -> None:
    callback = Autometrics(
        callback_config=_config(
            queue_name="pretrain-ml",
            priority_class="high",
            gpu_type="h100",
            extra_env={"FOO": "1"},
        )
    )

    assert callback.job_kwargs["queue_name"] == "pretrain-ml"
    assert callback.job_kwargs["priority_class"] == "high"
    assert callback.job_kwargs["gpu_type"] == "h100"
    assert callback.job_kwargs["extra_env"] == {"FOO": "1"}


def test_callback_own_keys_are_not_forwarded() -> None:
    callback = Autometrics(
        callback_config=_config(stable_save_interval=500, sglang_tag="custom")
    )

    assert not {"user", "nnodes", "eval_interval", "stable_save_interval"} & set(
        callback.job_kwargs
    )
    assert callback.job_kwargs["sglang_tag"] == "custom"


def test_defaults_are_overridable() -> None:
    assert Autometrics(callback_config=_config()).job_kwargs == {
        "max_retry": 3,
        "slurm_submit_type": "api",
    }
    assert (
        Autometrics(callback_config=_config(max_retry=0)).job_kwargs["max_retry"] == 0
    )
