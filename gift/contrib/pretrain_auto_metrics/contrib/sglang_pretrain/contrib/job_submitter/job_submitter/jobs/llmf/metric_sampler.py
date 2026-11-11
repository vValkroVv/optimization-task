"""The metric sampler of an llm-foundry training job.

The sampler records per-rank, per-microbatch training statistics that its collector carries to ClickHouse and
Grafana. A job configures it with a YAML file of the shape of `MetricSamplerConfig` -- a preset from
`metric_sampler_configs/` or a file of one's own -- and field overrides. `TrainJobLLMF` hands the result to the
training as overrides of its config: the `metric_sampler` callback and logger, and the monitors the monitor family
needs that the config lacks. `MetricSamplerLoad` estimates how many values a training step sends to the backend on
average, which `TrainJobAssertor` holds to what the backend takes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from omegaconf import OmegaConf

CONFIGS_DIR = Path(__file__).parent / "metric_sampler_configs"
PRESETS = ("light", "standard", "full")
# Hidden-width activation points the sampler can dump.
DUMP_POINTS = (
    "residual",
    "attn_out",
    "mlp_out",
    "final_norm",
    "shared_expert_out",
    "attn_residual_contribution",
    "mlp_residual_contribution",
    "embed",
)
# The values of an override pass through the startup script's heredoc and `bash -c` inside single quotes.
UNQUOTABLE = ("$", "`", "\\", "'")
# What the metric sampler backend takes on average per training step. The island's S3 proxy is the tighter limit: at
# the 20-25 s steps of large runs it carries ~1.0-1.25e9 values (~1.2 bytes each, compressed) per step, while the
# ClickHouse server that loads every island's windows has room for several times that. A 37b-like MoE model
# (34 layers, 192 experts, 8 microbatches, every family) records ~1e6 values per rank and step.
MAX_VALUES_PER_STEP = 1.25e9


@dataclass
class CaptureConfig:
    every_n_steps: int = 1
    distributions_every_n_steps: int = 100
    window_steps: int = 10


@dataclass
class FamiliesConfig:
    activations: bool = True
    token_loss: bool = True
    experts: bool = True
    attention: bool = False
    monitors: bool = True


@dataclass
class ActivationDumpConfig:
    every_n_steps: int = 0
    points: list[str] = field(default_factory=list)
    layers: list[int] = field(default_factory=list)
    tokens: int = 256
    max_gb: float = 10.0


@dataclass
class MetricSamplerConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    families: FamiliesConfig = field(default_factory=FamiliesConfig)
    exact_quantiles: bool = True
    activation_dump: ActivationDumpConfig = field(default_factory=ActivationDumpConfig)

    @classmethod
    def load(cls, source: str, overrides: dict) -> MetricSamplerConfig:
        """The config of the preset named `source`, or of the YAML file at `source`, with `overrides` (nested like
        the file) merged in. Unknown keys and values of the wrong type raise."""
        path = Path(source) if source.endswith((".yaml", ".yml")) else CONFIGS_DIR / f"{source}.yaml"
        merged = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.load(path), overrides)
        return OmegaConf.to_object(merged)

    def callback(self, output_dir: str) -> dict:
        """The arguments of the training's `metric_sampler` callback."""
        callback = {
            "output_dir": output_dir,
            "window_steps": self.capture.window_steps,
            "distribution_interval": self.capture.distributions_every_n_steps,
            "capture_interval": self.capture.every_n_steps,
            "activations": self.families.activations,
            "token_loss": self.families.token_loss,
            "experts": self.families.experts,
            "attention_stats": self.families.attention,
            "rank_local_monitors": self.families.monitors,
            "exact_quantiles": self.exact_quantiles,
        }
        dump = self.activation_dump
        if dump.every_n_steps > 0:
            callback["activation_dump"] = {
                "interval": dump.every_n_steps,
                "points": dump.points,
                "layers": dump.layers,
                "tokens": dump.tokens,
                "max_gb": dump.max_gb,
            }
        return callback

    def training_overrides(self, output_dir: str, training_config: dict) -> list[str]:
        """Overrides of the training config, as `key=value` arguments of the training script: the monitors the
        monitor family needs and `training_config` lacks, then the callback, which runs after them, and the logger."""
        overrides = {"callbacks": {}}
        if self.families.monitors:
            callbacks = training_config.get("callbacks") or {}
            for name, settings in required_monitors().items():
                if name not in callbacks:
                    overrides["callbacks"][name] = settings
        overrides["callbacks"]["metric_sampler"] = self.callback(output_dir)
        overrides["loggers"] = {
            "metric_sampler": {"output_dir": output_dir, "window_steps": self.capture.window_steps}
        }
        return [f"{key}={value}" for key, value in _dotlist(overrides)]


def required_monitors() -> dict[str, dict]:
    """The monitors the monitor family reads, with the settings it requires, in the order the callbacks run."""
    return OmegaConf.to_container(OmegaConf.load(CONFIGS_DIR / "monitors.yaml"))


def _dotlist(values: dict, prefix: str = "") -> list[tuple[str, str]]:
    items = []
    for key, value in values.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            items.extend(_dotlist(value, f"{dotted}."))
        else:
            items.append((dotted, json.dumps(value) if isinstance(value, (bool, dict, list)) else str(value)))
    return items


@dataclass(frozen=True)
class MetricSamplerLoad:
    """Values a training step sends to the metric sampler backend on average, from the training config's model and
    batch.

    The per-rank figures are the values per rank and step recorded on a 37b-like MoE run, scaled by the model:
    expert and monitor values grow with the routed layers, microbatches and experts, activation and attention
    values with the layers and microbatches, token-loss values with the microbatches and data streams. Steps with
    detailed distributions add the expert-loss and sample histograms."""

    world_size: int
    layers: int
    moe_layers: int
    experts: int
    microbatches: int
    streams: int

    # Measured values per rank-step and per unit of their scale (normal steps, 8-layer 192-expert stand run, which
    # the 37b run's 1.02M values per rank-step confirm within 8%).
    EXPERT_VALUES = 5.4  # per routed layer x microbatch x expert
    MONITOR_VALUES = 5.7  # per routed layer x microbatch x expert
    ACTIVATION_VALUES = 1141  # per layer x microbatch
    ATTENTION_VALUES = 650  # per layer x microbatch
    TOKEN_LOSS_VALUES_PER_STREAM = 260  # per microbatch x stream
    TOKEN_LOSS_VALUES = 400  # per microbatch
    FIXED_VALUES = 100
    EXPERT_LOSS_BINS = 64
    SAMPLE_LOSS_BINS = 128

    @classmethod
    def from_training_config(cls, training_config: dict, world_size: int) -> MetricSamplerLoad:
        model = dict(_pretrained_model_config(training_config))
        model.update((training_config.get("model") or {}).get("config_overrides") or {})
        layers = int(model.get("num_hidden_layers", 0))
        mtp_layers = int(model.get("mtp_predictor_num", 0)) if model.get("use_mtp") else 0
        microbatch = training_config.get("device_train_microbatch_size")
        global_batch = int(training_config.get("global_train_batch_size", 0))
        microbatches = math.ceil(global_batch / (world_size * microbatch)) if isinstance(microbatch, int) else 1
        streams = (training_config.get("train_loader") or {}).get("dataset", {}).get("streams") or {"all": None}
        return cls(
            world_size=world_size,
            layers=layers + mtp_layers,
            moe_layers=max(layers - int(model.get("first_k_dense_replace", 0)), 0),
            experts=int(model.get("num_routed_experts", 0)),
            microbatches=max(microbatches, 1),
            streams=len(streams),
        )

    def values_per_step(self, callback: dict) -> float:
        """Values per training step, on average, for the `metric_sampler` callback arguments."""
        moe = self.moe_layers * self.microbatches * self.experts
        per_rank = self.FIXED_VALUES
        detailed = 0.0
        if callback.get("experts", True):
            per_rank += self.EXPERT_VALUES * moe
            detailed += moe * (self.EXPERT_LOSS_BINS + 2 * self.streams)
        if callback.get("rank_local_monitors", False):
            per_rank += self.MONITOR_VALUES * moe
        if callback.get("activations", True):
            per_rank += self.ACTIVATION_VALUES * self.layers * self.microbatches
        if callback.get("attention_stats", False):
            per_rank += self.ATTENTION_VALUES * self.layers * self.microbatches
        if callback.get("token_loss", True):
            per_rank += self.microbatches * (self.TOKEN_LOSS_VALUES_PER_STREAM * self.streams + self.TOKEN_LOSS_VALUES)
            detailed += self.microbatches * self.streams * self.SAMPLE_LOSS_BINS
        capture_interval = callback.get("capture_interval", 1)
        detailed_share = capture_interval / callback.get("distribution_interval", 100)
        return self.world_size * (per_rank + detailed * detailed_share) / capture_interval


def _pretrained_model_config(training_config: dict) -> dict:
    """The pretrained model's config.json, which `model.config_overrides` override, or nothing without one."""
    path = (training_config.get("model") or {}).get("pretrained_model_name_or_path")
    config_file = Path(path) / "config.json" if path else None
    if config_file is None or not config_file.is_file():
        return {}
    return json.loads(config_file.read_text(encoding="utf-8"))


def output_dir(raw_root: str, training_config: dict) -> str:
    """Where the training writes the sampler's windows: the collector reads <raw root>/<experiment>/<run>."""
    return f"{raw_root}/{training_config.get('exp_name')}/{training_config.get('run_name')}"


def overridable(value: str) -> bool:
    return not any(character in value for character in UNQUOTABLE)
