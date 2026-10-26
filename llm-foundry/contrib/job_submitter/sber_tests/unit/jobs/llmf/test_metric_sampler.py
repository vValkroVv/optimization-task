import json
from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError, ValidationError

from job_submitter.jobs.llmf.metric_sampler import (
    MAX_VALUES_PER_STEP,
    PRESETS,
    MetricSamplerConfig,
    MetricSamplerLoad,
    output_dir,
    overridable,
    required_monitors,
)

pytestmark = pytest.mark.unit

# A 37b-like MoE model: 34 layers (the first 3 dense), 192 experts, an MTP layer, 8 microbatches on 256 GPUs and
# 44 data streams.
MODEL_37B = {
    "num_hidden_layers": 34,
    "num_routed_experts": 192,
    "first_k_dense_replace": 3,
    "use_mtp": True,
    "mtp_predictor_num": 1,
}


def training_config_37b(global_batch: int = 2048) -> dict:
    return {
        "model": {"config_overrides": dict(MODEL_37B)},
        "global_train_batch_size": global_batch,
        "device_train_microbatch_size": 1,
        "train_loader": {"dataset": {"streams": {f"stream{index}": {} for index in range(44)}}},
    }


@pytest.mark.parametrize(
    ("preset", "every_n_steps", "monitors", "attention", "exact_quantiles"),
    [("light", 10, False, False, False), ("standard", 1, True, False, True), ("full", 1, True, True, True)],
)
def test_presets_set_the_capture_and_the_families(preset, every_n_steps, monitors, attention, exact_quantiles):
    config = MetricSamplerConfig.load(preset, {})

    assert preset in PRESETS
    assert config.capture.every_n_steps == every_n_steps
    assert config.capture.distributions_every_n_steps == 100
    assert (config.families.monitors, config.families.attention) == (monitors, attention)
    assert config.families.activations and config.families.token_loss and config.families.experts
    assert config.exact_quantiles == exact_quantiles
    assert config.activation_dump.every_n_steps == 0


def test_overrides_and_a_file_of_ones_own_change_single_fields(tmp_path: Path):
    own = tmp_path / "sampler.yaml"
    own.write_text("capture:\n  every_n_steps: 4\nfamilies:\n  attention: true\n", encoding="utf-8")

    config = MetricSamplerConfig.load(str(own), {"families": {"experts": False}, "exact_quantiles": False})

    assert config.capture.every_n_steps == 4 and config.capture.window_steps == 10
    assert config.families.attention and not config.families.experts and not config.exact_quantiles


@pytest.mark.parametrize(
    ("overrides", "error"),
    [({"capture": {"every_n_step": 4}}, ConfigKeyError), ({"capture": {"every_n_steps": "often"}}, ValidationError)],
)
def test_an_unknown_field_or_a_value_of_the_wrong_type_is_refused(overrides, error):
    with pytest.raises(error):
        MetricSamplerConfig.load("standard", overrides)


def test_the_callback_takes_every_setting_and_a_dump_only_when_it_is_on():
    config = MetricSamplerConfig.load(
        "standard",
        {"capture": {"every_n_steps": 4}, "activation_dump": {"every_n_steps": 100, "points": ["residual"], "layers": [0]}},
    )

    assert config.callback("/raw/exp/run") == {
        "output_dir": "/raw/exp/run",
        "window_steps": 10,
        "distribution_interval": 100,
        "capture_interval": 4,
        "activations": True,
        "token_loss": True,
        "experts": True,
        "attention_stats": False,
        "rank_local_monitors": True,
        "exact_quantiles": True,
        "activation_dump": {"interval": 100, "points": ["residual"], "layers": [0], "tokens": 256, "max_gb": 10.0},
    }
    assert "activation_dump" not in MetricSamplerConfig.load("standard", {}).callback("/raw/exp/run")


def test_training_overrides_add_the_missing_monitors_then_the_callback_and_the_logger():
    config = MetricSamplerConfig.load("standard", {})

    overrides = config.training_overrides("/raw/exp/run", {"callbacks": {"lr_monitor": {}, "weight_monitor": {}}})

    keys = [override.split("=", 1)[0] for override in overrides]
    monitors = [name for name in required_monitors() if name not in ("lr_monitor", "weight_monitor")]
    first_of = [next(index for index, key in enumerate(keys) if key.startswith(f"callbacks.{name}")) for name in monitors]
    assert first_of == sorted(first_of) and first_of[-1] < keys.index("callbacks.metric_sampler.output_dir")
    assert not any(key.startswith(("callbacks.lr_monitor", "callbacks.weight_monitor")) for key in keys)
    assert "callbacks.memory_monitor={}" in overrides
    assert 'callbacks.topk_accuracy_monitor.k_values=[1, 5, 20]' in overrides
    assert 'callbacks.activation_monitor.stat_list=["amax", "l2norm"]' in overrides
    assert "callbacks.metric_sampler.rank_local_monitors=true" in overrides
    assert overrides[-2:] == ["loggers.metric_sampler.output_dir=/raw/exp/run", "loggers.metric_sampler.window_steps=10"]


def test_without_the_monitor_family_no_monitor_is_added():
    overrides = MetricSamplerConfig.load("light", {}).training_overrides("/raw/exp/run", {})

    assert all(override.startswith(("callbacks.metric_sampler.", "loggers.metric_sampler.")) for override in overrides)
    assert "callbacks.metric_sampler.capture_interval=10" in overrides


def test_the_load_of_a_37b_like_run_is_the_values_per_rank_it_recorded():
    standard = MetricSamplerConfig.load("standard", {}).callback("/x")

    load = MetricSamplerLoad.from_training_config(training_config_37b(), world_size=256)

    assert (load.layers, load.moe_layers, load.experts, load.microbatches, load.streams) == (35, 31, 192, 8, 44)
    assert load.values_per_step(standard) / 256 == pytest.approx(1.02e6, rel=0.02)


def test_the_load_follows_the_capture_interval_and_the_families():
    load = MetricSamplerLoad.from_training_config(training_config_37b(global_batch=8 * 1000), world_size=1000)
    standard = MetricSamplerConfig.load("standard", {}).callback("/x")
    every = load.values_per_step(standard)

    assert every < MAX_VALUES_PER_STEP < 2 * every
    # Detailed steps come every 100 steps whatever the interval: only the other steps' values shrink fourfold.
    assert every / 4 < load.values_per_step({**standard, "capture_interval": 4}) < every / 2
    lighter = {**standard, "experts": False, "rank_local_monitors": False}
    assert load.values_per_step(lighter) < every / 2
    with_attention = load.values_per_step({**standard, "attention_stats": True})
    without_any = load.values_per_step(
        {**standard, "experts": False, "rank_local_monitors": False, "activations": False, "token_loss": False})
    assert with_attention > every and without_any == pytest.approx(1000 * MetricSamplerLoad.FIXED_VALUES)


def test_the_model_shape_comes_from_the_pretrained_config_under_the_overrides(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({**MODEL_37B, "num_routed_experts": 64}), encoding="utf-8")
    config = {
        "model": {"pretrained_model_name_or_path": str(tmp_path), "config_overrides": {"num_hidden_layers": 8}},
        "global_train_batch_size": 16,
        "device_train_microbatch_size": 1,
    }

    load = MetricSamplerLoad.from_training_config(config, world_size=16)

    assert (load.layers, load.moe_layers, load.experts, load.microbatches, load.streams) == (9, 5, 64, 1, 1)


@pytest.mark.parametrize(
    "config",
    [{}, {"model": {"pretrained_model_name_or_path": "/no/such/model"}, "device_train_microbatch_size": "auto"}],
    ids=["nothing known", "missing pretrained config and an automatic microbatch"],
)
def test_an_unknown_model_has_no_layers_or_experts(config):
    load = MetricSamplerLoad.from_training_config(config, world_size=8)

    assert (load.layers, load.experts, load.microbatches, load.streams) == (0, 0, 1, 1)


def test_the_output_directory_is_the_one_the_collector_reads_and_overrides_stay_quotable():
    assert output_dir("/raw", {"exp_name": "exp", "run_name": "run"}) == "/raw/exp/run"
    assert overridable("callbacks.metric_sampler.output_dir=/raw/exp/run")
    assert not any(overridable(f"key={value}") for value in ("$HOME", "`id`", "a\\b", "it's"))
