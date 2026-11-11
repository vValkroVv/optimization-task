from __future__ import annotations

import shlex
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from job_submitter.asserts.base import BaseJobAssertor
from job_submitter.core.job_config import JobTemplate
from job_submitter.core.job_templates.region_configs import METRIC_SAMPLER_RAW_ROOTS
from job_submitter.jobs.llmf.autometrics import AutometricsJobLLMF
from job_submitter.jobs.llmf.metric_sampler import (
    DUMP_POINTS,
    MAX_VALUES_PER_STEP,
    MetricSamplerLoad,
    output_dir,
    overridable,
    required_monitors,
)

if TYPE_CHECKING:
    from job_submitter.jobs.llmf.train import TrainJobLLMF

CALLBACK_OWN_KEYS = frozenset(
    {"eval_interval", "stable_save_interval", "nnodes", "user"}
)
CALLBACK_SET_JOB_KEYS = frozenset(
    {"checkpoint_dir", "user_login", "exp_name", "n_workers"}
)
AUTOMETRICS_CALLBACK_KEYS = (
    frozenset(field.name for field in fields(AutometricsJobLLMF) if field.init)
    - CALLBACK_SET_JOB_KEYS
) | CALLBACK_OWN_KEYS
METRIC_SAMPLER_CALLBACK_KEYS = frozenset(
    {
        "output_dir",
        "window_steps",
        "distribution_interval",
        "capture_interval",
        "activations",
        "token_loss",
        "experts",
        "attention_stats",
        "rank_local_monitors",
        "exact_quantiles",
        "activation_dump",
    }
)
# Monitor settings the monitor family reads as they are: (monitor, setting, the value it needs, its default).
METRIC_SAMPLER_MONITOR_SETTINGS = (
    ("weight_monitor", "stat_list", ["amax", "l2norm"], None),
    ("activation_monitor", "stat_list", ["amax", "l2norm"], None),
    ("topk_accuracy_monitor", "k_values", [1, 5, 20], [1, 5, 20]),
    ("optimizer_monitor", "log_pre_clip_gradients", False, False),
    ("memory_monitor", "memory_keys", None, None),
)


class TrainJobAssertor(BaseJobAssertor):
    def _assert_autometrics_callback_keys(self, autometrics: dict) -> None:
        unknown = sorted(set(autometrics) - AUTOMETRICS_CALLBACK_KEYS)
        assert not unknown, self._pretty_config_param_print(
            {"callbacks.autometrics": sorted(AUTOMETRICS_CALLBACK_KEYS)},
            f"unknown keys {unknown}, every key must be one of",
        )

    def _assert_autometrics_config(self, train_config: dict, config_path: Path) -> None:
        callback_config = train_config.get("callbacks", {}).get("autometrics")
        if callback_config is not None:
            self._assert_autometrics_callback_keys(callback_config)

        if "autometrics" in train_config:
            assert "callbacks" in train_config, self._pretty_print_missing_key(
                "callbacks"
            )
            callbacks = train_config["callbacks"]
            assert "autometrics" in callbacks, self._pretty_print_missing_key(
                "callbacks.autometrics"
            )
            autometrics = callbacks["autometrics"]
            assert "eval_interval" in autometrics, self._pretty_print_missing_key(
                "autometrics.eval_interval"
            )
            assert "stable_save_interval" in autometrics, (
                self._pretty_print_missing_key("autometrics.stable_save_interval")
            )
            assert "nnodes" in autometrics, self._pretty_print_missing_key(
                "autometrics.nnodes"
            )
            assert "user" in autometrics, self._pretty_print_missing_key(
                "autometrics.user"
            )
        elif "callbacks" in train_config and "autometrics" in train_config["callbacks"]:
            autometrics = train_config["callbacks"]["autometrics"]
            assert "eval_interval" in autometrics, self._pretty_print_missing_key(
                "autometrics.eval_interval"
            )
            assert "stable_save_interval" in autometrics, (
                self._pretty_print_missing_key("autometrics.stable_save_interval")
            )
            assert "nnodes" in autometrics, self._pretty_print_missing_key(
                "autometrics.nnodes"
            )
            assert "user" in autometrics, self._pretty_print_missing_key(
                "autometrics.user"
            )
            assert "autometrics" in train_config, self._pretty_print_missing_key(
                "autometrics"
            )

        if (
            "autometrics" in train_config
            and "callbacks" in train_config
            and "autometrics" in train_config["callbacks"]
        ):
            if "sglang_tag" in train_config["callbacks"]["autometrics"]:
                assert train_config["callbacks"]["autometrics"]["sglang_tag"] in (
                    "opensource",
                    "custom",
                ), self._pretty_config_param_print(
                    {
                        "autometrics.callbacks.autometrics.sglang_tag": [
                            "opensource",
                            "custom",
                        ]
                    }
                )
            job = AutometricsJobLLMF(
                user_login=train_config["user"],
                exp_name="mock_exp",
                n_workers=train_config["callbacks"]["autometrics"]["nnodes"],
                n_gpus=train_config["callbacks"]["autometrics"].get("n_gpus", 8),
                slurm_submit_type="api",
                path_to_config=str(config_path),
                checkpoint_dir=".",
            )
            job_config = job.to_config()
            job.assert_job(job, job_config, source_job="TrainJobLLMF")

    def _assert_metric_sampler(self, job: TrainJobLLMF, train_config: dict) -> None:
        """The metric sampler of the training config llm-foundry will see -- the file with the job's metric sampler
        overrides and script_args -- set up so that it works, writes where its collector reads, and stays within
        what the backend takes per step unless the job says why it may not."""
        sampler_overrides = job.metric_sampler_training_overrides()
        if job.metric_sampler is not None:
            for section in ("callbacks", "loggers"):
                assert "metric_sampler" not in (train_config.get(section) or {}), (
                    f"{section}.metric_sampler of {job.path_to_config} would set up the metric sampler a second "
                    f"time: remove it, TrainJobLLMF(metric_sampler={job.metric_sampler!r}) sets it up"
                )
        for override in sampler_overrides:
            assert overridable(override), f"metric sampler setting {override!r} cannot hold $, `, \\ or '"
        effective = OmegaConf.to_container(
            OmegaConf.merge(
                OmegaConf.create(train_config),
                OmegaConf.from_cli(sampler_overrides + shlex.split(job.script_args)),
            )
        )
        callbacks = effective.get("callbacks") or {}
        callback = callbacks.get("metric_sampler")
        if callback is None:
            return

        assert job.region in METRIC_SAMPLER_RAW_ROOTS, self._pretty_params_print(
            {"region": sorted(METRIC_SAMPLER_RAW_ROOTS)}
        )
        unknown = sorted(set(callback) - METRIC_SAMPLER_CALLBACK_KEYS)
        assert not unknown, self._pretty_config_param_print(
            {"callbacks.metric_sampler": sorted(METRIC_SAMPLER_CALLBACK_KEYS)},
            f"unknown keys {unknown}, every key must be one of",
        )
        for name in ("exp_name", "run_name"):
            assert str(effective.get(name) or "").strip() and "/" not in str(effective[name]), (
                self._pretty_print_missing_key(name) + " (a name without /: the metric sampler writes into it)"
            )
        expected_dir = output_dir(METRIC_SAMPLER_RAW_ROOTS[job.region], effective)
        assert callback.get("output_dir") == expected_dir, self._pretty_config_param_print(
            {"callbacks.metric_sampler.output_dir": [expected_dir]},
            "the metric sampler collector reads <raw root>/<exp_name>/<run_name>",
        )
        capture_interval = callback.get("capture_interval", 1)
        for key, default in (("window_steps", 10), ("distribution_interval", 100), ("capture_interval", 1)):
            value = callback.get(key, default)
            assert isinstance(value, int) and not isinstance(value, bool) and value > 0, (
                self._pretty_config_param_print({f"callbacks.metric_sampler.{key}": ["a positive integer"]})
            )
        assert callback.get("distribution_interval", 100) % capture_interval == 0, self._pretty_config_param_print(
            {"callbacks.metric_sampler.distribution_interval": [f"a multiple of capture_interval={capture_interval}"]},
            "detailed distributions are recorded on captured steps",
        )
        self._assert_metric_sampler_dump(callback.get("activation_dump") or {}, callback, capture_interval)
        self._assert_metric_sampler_logger(effective.get("loggers") or {}, callback)
        if callback.get("rank_local_monitors", False):
            self._assert_metric_sampler_monitors(callbacks)
        self._assert_metric_sampler_training(effective)
        self._assert_metric_sampler_capacity(job, effective, callback)

    def _assert_metric_sampler_dump(self, dump: dict, callback: dict, capture_interval: int) -> None:
        interval = dump.get("interval", 0)
        if interval == 0:
            return
        assert interval > 0 and interval % capture_interval == 0, self._pretty_config_param_print(
            {"callbacks.metric_sampler.activation_dump.interval": [f"a multiple of capture_interval={capture_interval}"]}
        )
        assert callback.get("activations", True), (
            "activation dumps need the activation family: set families.activations (callbacks.metric_sampler."
            "activations) to true or turn the dumps off"
        )
        points = dump.get("points") or []
        assert points and set(points) <= set(DUMP_POINTS), self._pretty_config_param_print(
            {"callbacks.metric_sampler.activation_dump.points": list(DUMP_POINTS)}, "a non-empty list of"
        )
        assert dump.get("layers"), self._pretty_print_missing_key("callbacks.metric_sampler.activation_dump.layers")
        assert dump.get("tokens", 256) > 0 and dump.get("max_gb", 10.0) > 0, self._pretty_config_param_print(
            {"callbacks.metric_sampler.activation_dump": ["tokens > 0 and max_gb > 0"]}
        )

    def _assert_metric_sampler_logger(self, loggers: dict, callback: dict) -> None:
        logger = loggers.get("metric_sampler")
        if logger is None:
            return
        assert logger.get("output_dir") == callback["output_dir"], self._pretty_config_param_print(
            {"loggers.metric_sampler.output_dir": [callback["output_dir"]]}, "the callback's output_dir"
        )
        names = list(loggers)
        tensorboard = loggers.get("tensorboard") or {}
        assert tensorboard.get("rank_zero_only") and names.index("tensorboard") < names.index("metric_sampler"), (
            self._pretty_config_param_print(
                {"loggers.tensorboard.rank_zero_only": [True]},
                "listed before loggers.metric_sampler, which reads the scalars it logs",
            )
        )

    def _assert_metric_sampler_monitors(self, callbacks: dict) -> None:
        names = list(callbacks)
        sampler_index = names.index("metric_sampler")
        for monitor in required_monitors():
            assert monitor in callbacks and names.index(monitor) < sampler_index, (
                f"the metric sampler's monitor family needs callbacks.{monitor} listed before callbacks."
                f"metric_sampler, which runs last (or set families.monitors to false)"
            )
        for monitor, setting, needed, default in METRIC_SAMPLER_MONITOR_SETTINGS:
            assert (callbacks[monitor] or {}).get(setting, default) == needed, self._pretty_config_param_print(
                {f"callbacks.{monitor}.{setting}": [needed]}, "the metric sampler's monitor family reads it as"
            )

    def _assert_metric_sampler_training(self, config: dict) -> None:
        for key in ("tp_size", "sp_size"):
            assert config.get(key, 1) == 1, self._pretty_config_param_print(
                {key: [1]}, "the metric sampler records whole layers on each rank"
            )
        assert isinstance(config.get("device_train_microbatch_size"), int), self._pretty_config_param_print(
            {"device_train_microbatch_size": ["a fixed integer"]}, "the metric sampler lays out microbatches by it"
        )
        assert config.get("precision") != "amp_fp16", self._pretty_config_param_print(
            {"precision": ["amp_bf16", "fp32"]}, "the metric sampler cannot read gradients scaled by amp_fp16"
        )
        zero_stage = (config.get("gigafsdp_config") or {}).get("zero_stage", 1)
        assert zero_stage == 1, self._pretty_config_param_print(
            {"gigafsdp_config.zero_stage": [1]}, "the metric sampler reads gradients before their reduction"
        )

    def _assert_metric_sampler_capacity(self, job: TrainJobLLMF, config: dict, callback: dict) -> None:
        world_size = job.n_workers * job.processes_per_worker
        load = MetricSamplerLoad.from_training_config(config, world_size)
        assert load.experts > 0 and load.layers > 0, (
            "the metric sampler records MoE models: model.config_overrides (or the pretrained config.json) needs "
            "num_hidden_layers and num_routed_experts"
        )
        values = load.values_per_step(callback)
        capture_interval = callback.get("capture_interval", 1)
        share = values / MAX_VALUES_PER_STEP
        print(
            f"[INFO] metric sampler: every {capture_interval} step(s) on {world_size} GPUs, ~{values:.3g} values per "
            f"step, {share:.0%} of what the backend takes ({MAX_VALUES_PER_STEP:.3g})"
        )
        if share <= 1:
            return
        needed = capture_interval * -(-values // MAX_VALUES_PER_STEP)
        message = (
            f"the metric sampler would send ~{values:.3g} values per step, {share:.1f}x what its backend takes "
            f"({MAX_VALUES_PER_STEP:.3g}): capture every "
            f">= {int(needed)} steps (capture.every_n_steps), or switch families off"
        )
        if job.metric_sampler_unsafe_reason:
            print(f"[WARNING] {message}; going ahead because: {job.metric_sampler_unsafe_reason}")
            return
        raise AssertionError(
            f"{message}. A run that must go beyond it (a short debugging run, say) says why in "
            f"metric_sampler_unsafe_reason"
        )

    def assert_job(self, job: TrainJobLLMF, job_template: JobTemplate) -> None:
        super().assert_job(job, job_template)
        config = Path(job.path_to_config)
        assert config.exists(), self._pretty_params_print(
            {"path_to_config": ["/some/valid/train/config.yaml"]}
        )
        loaded_config = self._load_config(config)
        self._assert_autometrics_config(loaded_config, config)
        self._assert_metric_sampler(job, loaded_config)
