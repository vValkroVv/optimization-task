from pathlib import Path

import pytest
import yaml

from job_submitter.asserts.train import TrainJobAssertor
from job_submitter.jobs.llmf.metric_sampler import required_monitors
from job_submitter.jobs.llmf.train import TrainJobLLMF

pytestmark = pytest.mark.unit

RAW_ROOT = "/mnt/nfs/07-share/optimusprime/metric-sampler/node"
OUTPUT_DIR = f"{RAW_ROOT}/exp/run"


def training_config(**changes) -> dict:
    """A small MoE training config the metric sampler can record: 8 layers, 192 experts, 16 microbatches on 16 GPUs."""
    config = {
        "exp_name": "exp",
        "run_name": "run",
        "tp_size": 1,
        "precision": "amp_bf16",
        "global_train_batch_size": 256,
        "device_train_microbatch_size": 1,
        "gigafsdp_config": {"zero_stage": 1},
        "model": {"config_overrides": {"num_hidden_layers": 8, "num_routed_experts": 192, "first_k_dense_replace": 1}},
        "callbacks": {"lr_monitor": {}},
        "loggers": {"tensorboard": {"log_dir": "/tensorboard", "rank_zero_only": True}},
    }
    config.update(changes)
    return config


def job(tmp_path: Path, config: dict, **fields) -> TrainJobLLMF:
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return TrainJobLLMF(
        user_login="u", exp_name="e", submission_type="slurm", path_to_config=path,
        **{"region": "o1", "n_workers": 2, **fields},
    )


def check(sampler_job: TrainJobLLMF) -> None:
    TrainJobAssertor()._assert_metric_sampler(sampler_job, TrainJobAssertor._load_config(Path(sampler_job.path_to_config)))


def manual_callbacks(**sampler) -> dict:
    """callbacks of a training config that sets the sampler up by hand, with the monitor family's monitors."""
    callbacks = dict(required_monitors())
    callbacks["metric_sampler"] = {"output_dir": OUTPUT_DIR, "rank_local_monitors": True, **sampler}
    return callbacks


@pytest.mark.parametrize("preset", ["light", "standard", "full"])
def test_a_preset_passes_and_reports_its_load(tmp_path, capsys, preset):
    check(job(tmp_path, training_config(), metric_sampler=preset))

    assert "[INFO] metric sampler: every" in capsys.readouterr().out


def test_a_config_without_the_sampler_is_left_alone(tmp_path, capsys):
    check(job(tmp_path, training_config(tp_size=2)))

    assert capsys.readouterr().out == ""


def test_a_sampler_set_up_by_hand_is_held_to_the_same_rules(tmp_path):
    check(job(tmp_path, training_config(callbacks=manual_callbacks(capture_interval=4, window_steps=5))))
    with pytest.raises(AssertionError, match="distribution_interval"):
        check(job(tmp_path, training_config(callbacks=manual_callbacks(capture_interval=3))))


@pytest.mark.parametrize("section", ["callbacks", "loggers"])
def test_a_preset_refuses_a_sampler_the_training_config_sets_up_too(tmp_path, section):
    config = training_config()
    config[section]["metric_sampler"] = {"output_dir": OUTPUT_DIR}

    with pytest.raises(AssertionError, match=f"{section}.metric_sampler of .* second time"):
        check(job(tmp_path, config, metric_sampler="standard"))


def test_the_sampler_needs_an_island_with_its_collector(tmp_path):
    with pytest.raises(AssertionError, match="region"):
        job(tmp_path, training_config(), metric_sampler="standard", region="o2").metric_sampler_training_overrides()
    config = training_config(callbacks={"metric_sampler": {"output_dir": OUTPUT_DIR}})
    with pytest.raises(AssertionError, match="region"):
        check(job(tmp_path, config, region="o2"))


def test_an_unknown_preset_is_refused_with_the_known_ones(tmp_path):
    with pytest.raises(AssertionError, match="standard"):
        job(tmp_path, training_config(), metric_sampler="everything").metric_sampler_training_overrides()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"run_name": "a/b"}, "run_name"),
        ({"exp_name": ""}, "exp_name"),
        ({"run_name": "$HOME"}, "cannot hold"),
    ],
)
def test_the_names_the_sampler_writes_under_must_be_plain(tmp_path, changes, message):
    with pytest.raises(AssertionError, match=message):
        check(job(tmp_path, training_config(**changes), metric_sampler="standard"))


@pytest.mark.parametrize(
    ("sampler", "message"),
    [
        ({"interval": 1}, "unknown keys"),
        ({"output_dir": "/elsewhere"}, "collector reads"),
        ({"window_steps": 0}, "window_steps"),
        ({"capture_interval": True}, "capture_interval"),
        ({"activation_dump": {"interval": 50, "points": ["residual"], "layers": [0]}, "capture_interval": 4}, "interval"),
        ({"activation_dump": {"interval": 50, "points": ["residual"], "layers": [0]}, "activations": False}, "activation family"),
        ({"activation_dump": {"interval": 50, "points": ["swiglu_in"], "layers": [0]}}, "points"),
        ({"activation_dump": {"interval": 50, "points": ["residual"]}}, "layers"),
        ({"activation_dump": {"interval": 50, "points": ["residual"], "layers": [0], "tokens": 0}}, "tokens"),
    ],
)
def test_a_callback_setting_that_cannot_work_is_refused(tmp_path, sampler, message):
    callbacks = manual_callbacks()
    callbacks["metric_sampler"].update(sampler)

    with pytest.raises(AssertionError, match=message):
        check(job(tmp_path, training_config(callbacks=callbacks)))


def test_a_dump_that_is_off_needs_nothing_else(tmp_path):
    check(job(tmp_path, training_config(callbacks=manual_callbacks(activation_dump={"interval": 0}))))


@pytest.mark.parametrize(
    ("loggers", "message"),
    [
        ({"tensorboard": {"rank_zero_only": True}, "metric_sampler": {"output_dir": "/elsewhere"}}, "output_dir"),
        ({"metric_sampler": {"output_dir": OUTPUT_DIR}, "tensorboard": {"rank_zero_only": True}}, "rank_zero_only"),
        ({"tensorboard": {"rank_zero_only": False}, "metric_sampler": {"output_dir": OUTPUT_DIR}}, "rank_zero_only"),
    ],
)
def test_the_logger_writes_with_the_callback_after_a_rank_zero_tensorboard(tmp_path, loggers, message):
    with pytest.raises(AssertionError, match=message):
        check(job(tmp_path, training_config(callbacks=manual_callbacks(), loggers=loggers)))


def test_the_monitor_family_needs_its_monitors_before_the_callback(tmp_path):
    callbacks = manual_callbacks()
    del callbacks["router_margin_monitor"]
    with pytest.raises(AssertionError, match="router_margin_monitor listed before"):
        check(job(tmp_path, training_config(callbacks=callbacks)))

    late = {"metric_sampler": manual_callbacks()["metric_sampler"], **required_monitors()}
    with pytest.raises(AssertionError, match="listed before"):
        check(job(tmp_path, training_config(callbacks=late)))


@pytest.mark.parametrize(
    ("monitor", "setting", "value"),
    [
        ("weight_monitor", "stat_list", ["amax"]),
        ("topk_accuracy_monitor", "k_values", [1, 5]),
        ("optimizer_monitor", "log_pre_clip_gradients", True),
        ("memory_monitor", "memory_keys", ["allocated_mem"]),
    ],
)
def test_a_monitor_setting_the_monitor_family_reads_differently_is_refused(tmp_path, monitor, setting, value):
    callbacks = manual_callbacks()
    callbacks[monitor] = {**(callbacks[monitor] or {}), setting: value}

    with pytest.raises(AssertionError, match=f"{monitor}"):
        check(job(tmp_path, training_config(callbacks=callbacks)))


def test_a_preset_adds_the_monitors_but_keeps_a_monitor_the_config_has(tmp_path):
    config = training_config(callbacks={"weight_monitor": {"stat_list": ["amax"]}})

    with pytest.raises(AssertionError, match="weight_monitor"):
        check(job(tmp_path, config, metric_sampler="standard"))
    check(job(tmp_path, config, metric_sampler="light"))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tp_size": 2}, "tp_size"),
        ({"sp_size": 2}, "sp_size"),
        ({"device_train_microbatch_size": "auto"}, "device_train_microbatch_size"),
        ({"precision": "amp_fp16"}, "precision"),
        ({"gigafsdp_config": {"zero_stage": 3}}, "zero_stage"),
    ],
)
def test_training_settings_the_sampler_cannot_record_are_refused(tmp_path, changes, message):
    with pytest.raises(AssertionError, match=message):
        check(job(tmp_path, training_config(**changes), metric_sampler="standard"))


def test_a_dense_model_is_refused(tmp_path):
    config = training_config(model={"config_overrides": {"num_hidden_layers": 8}})

    with pytest.raises(AssertionError, match="MoE"):
        check(job(tmp_path, config, metric_sampler="standard"))


def large_run(tmp_path: Path, **fields) -> TrainJobLLMF:
    """2048 GPUs of a 37b-like model with 8 microbatches each: about twice what the backend takes every step."""
    config = training_config(global_train_batch_size=8 * 2048)
    config["model"]["config_overrides"] = {"num_hidden_layers": 34, "num_routed_experts": 192, "first_k_dense_replace": 3}
    return job(tmp_path, config, metric_sampler="standard", n_workers=256, **fields)


def test_every_step_of_a_2k_gpu_run_is_refused_with_the_interval_that_fits(tmp_path):
    with pytest.raises(AssertionError, match=r"capture every >= 2 steps.*metric_sampler_unsafe_reason"):
        check(large_run(tmp_path))
    check(large_run(tmp_path, metric_sampler_overrides={"capture": {"every_n_steps": 2}}))


def test_a_reason_lets_a_run_beyond_the_backend_go_ahead_with_a_warning(tmp_path, capsys):
    check(large_run(tmp_path, metric_sampler_unsafe_reason="debugging a loss spike for 50 steps"))

    out = capsys.readouterr().out
    assert "[WARNING]" in out and "going ahead because: debugging a loss spike for 50 steps" in out


def test_script_args_are_part_of_the_config_that_is_checked(tmp_path):
    with pytest.raises(AssertionError, match="tp_size"):
        check(job(tmp_path, training_config(), metric_sampler="standard", script_args="tp_size=2"))
