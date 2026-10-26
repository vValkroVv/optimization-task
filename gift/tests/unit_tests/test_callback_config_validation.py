"""Local callback and validator errors must fail without building runtime objects."""

import pytest
import yaml

from torchtitan.config.decode import decode
from torchtitan.config.errors import ConfigFieldError, ConfigSchemaError
from torchtitan.config.schema.callbacks.builtin.autometrics import (
    AutometricsConfig,
)
from torchtitan.config.schema.callbacks.sources.modelling_source import (
    ModellingSourceConfig,
)
from torchtitan.config.schema.components.validate import (
    BaseValidatorConfig,
    ValidatorConfig,
)
from torchtitan.config.schema.models.flux.validate import FluxValidatorConfig
from torchtitan.config.schema.callbacks.metrics.sample import StatOpConfig


@pytest.mark.parametrize(
    "config_class,freq",
    [
        (BaseValidatorConfig, 0),
        (ValidatorConfig, -1),
        (FluxValidatorConfig, True),
        (ValidatorConfig, 1.5),
    ],
)
def test_validator_frequency_must_be_positive_integer(config_class, freq):
    with pytest.raises(ConfigFieldError) as exc:
        config_class(freq=freq)
    assert exc.value.field == "freq"


@pytest.mark.parametrize("freq", [1, 10])
def test_validator_accepts_valid_frequency(freq):
    assert decode(ValidatorConfig, {"enable": True, "freq": freq}).freq == freq


def test_unknown_stat_is_rejected_before_runtime():
    with pytest.raises(ConfigSchemaError, match="abs_maxx"):
        decode(StatOpConfig, {"op": "abs_maxx"})


def _autometrics(**overrides):
    data = dict(user="user", submission_type="cloud", tasks=[{"name": "benchmark"}])
    return decode(AutometricsConfig, data | overrides)


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"tasks": []}, "tasks"),
        ({"tasks": [{"name": "x"}, {"name": "x"}]}, "tasks"),
        ({"interval": 0}, "interval"),
        ({"inference_tp": 0}, "inference_tp"),
        ({"inference_dp": -1}, "inference_dp"),
        ({"n_workers": 0}, "n_workers"),
        ({"n_gpus": 0}, "n_gpus"),
        ({"inference_ep": -1}, "inference_ep"),
        ({"inference_ep": 2}, "inference_ep"),
        ({"inference_tp": 2}, "inference_dp"),
        ({"submission_type": ""}, "submission_type"),
        ({"submission_type": "other"}, "submission_type"),
        ({"max_retry": 1}, "max_retry"),
    ],
)
def test_autometrics_rejects_local_errors_before_runtime(overrides, field):
    with pytest.raises(ConfigSchemaError, match=field):
        _autometrics(**overrides)


@pytest.mark.parametrize("submission_type", ["cloud", "slurm"])
def test_autometrics_accepts_consistent_topology(submission_type):
    config = _autometrics(
        submission_type=submission_type,
        inference_tp=2,
        inference_dp=4,
        inference_ep=2,
        n_workers=2,
        n_gpus=4,
    )
    assert config.inference_tp == config.inference_ep


@pytest.mark.parametrize("mode", ["convertation", "inference"])
def test_autometrics_allows_no_benchmarks_in_other_modes(mode):
    assert not _autometrics(mode=mode, tasks=[]).tasks


@pytest.mark.parametrize(
    "plan, valid",
    [
        ("pipeline: [{op: abs_max}], reduce: {group: dp, op: max}", True),
        (
            "steps: [{kind: stat, stat: {op: abs_max}}, "
            "{kind: reduce, reduce: {group: dp, op: max}}]",
            True,
        ),
        ("pipeline: [{op: abs_max}]", False),
        (
            "steps: [{kind: stat, stat: {op: abs_max}}], reduce: {group: dp, op: max}",
            False,
        ),
        ("steps: [{kind: reduce, reduce: {}}]", False),
        ("steps: [{kind: reduce, reduce: {group: dp, op: none}}]", False),
        ("reduce: {stages: [{}]}", False),
        ("steps: [{kind: reduce, reduce: {group: []}}]", False),
        ("reduce: {stages: [{group: dp, op: none}]}", False),
    ],
)
def test_modelling_checks_the_executed_reduction_plan(plan, valid):
    data = {"metrics": [{"name": "x", **yaml.safe_load("{" + plan + "}")}]}
    if valid:
        assert decode(ModellingSourceConfig, data).metrics[0].name == "x"
    else:
        with pytest.raises(ConfigSchemaError, match="declare no reduce stages"):
            decode(ModellingSourceConfig, data)
