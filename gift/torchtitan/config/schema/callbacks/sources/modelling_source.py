from dataclasses import dataclass, field
from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.callbacks.metrics.sample import (
    MetricConfig,
    MetricStepKind,
    ReduceConfig,
    ReduceOp,
    ReduceStageConfig,
    StatOpConfig,
)
from torchtitan.config.schema.callbacks.sources.base import FilteredSourceConfig
from torchtitan.config.schema.protocols.types import StrEnum


class CaptureOutput(StrEnum):
    first_tensor = "first_tensor"
    all_tensors = "all_tensors"


def _default_tensor_metrics() -> list[MetricConfig]:
    def reduce_over_data(op: ReduceOp) -> ReduceConfig:
        return ReduceConfig(
            stages=[
                ReduceStageConfig(group="dp", op=op),
                ReduceStageConfig(group="cp", op=op),
            ]
        )

    return [
        MetricConfig(
            name="amax",
            pipeline=[StatOpConfig(op="abs_max")],
            reduce=reduce_over_data(ReduceOp.max),
        ),
        MetricConfig(
            name="amean",
            pipeline=[StatOpConfig(op="abs"), StatOpConfig(op="mean")],
            reduce=reduce_over_data(ReduceOp.mean),
        ),
        MetricConfig(
            name="l2norm",
            pipeline=[StatOpConfig(op="norm", dim=-1), StatOpConfig(op="mean")],
            reduce=reduce_over_data(ReduceOp.mean),
        ),
    ]


@dataclass(kw_only=True, slots=True)
class MoECaptureConfig:
    """Router load-balance and expert-weight statistics of every MoE layer.

    No ``enable``: like a callback absent from the run's dict, a block that
    asks for nothing is not collected -- see :attr:`requested`.
    """

    # Over the batch-summed (E,) token histogram.
    token_metrics: list[MetricConfig] = field(default_factory=list)
    # Over the batch-averaged (E,) per-expert mean router score.
    score_metrics: list[MetricConfig] = field(default_factory=list)
    bias_metrics: list[MetricConfig] = field(default_factory=list)
    # L1-normalized token fraction per expert: one series per expert per layer.
    per_expert_tokens: bool = False
    expert_weights: bool = False
    # Over each expert parameter, reduced across every axis but dim 0 (the
    # expert index), so each yields an (E,) vector published as one tag per
    # expert under ``{metric.name}/{unit}.{param}__expert_idx_{e}``.
    weight_stats: list[MetricConfig] = field(default_factory=list)
    # Over the per-expert l2-norm vector, one scalar each, published under
    # ``{namespace}/{unit}/expert_weights/{param}_{metric.name}``.
    weight_spread: list[MetricConfig] = field(default_factory=list)
    # Which of ``weight_stats`` produces the norm vector the spread metrics
    # summarise.
    norm_metric_name: str = "weights_l2norm"
    # Tag prefix of every MoE series.
    namespace: str = "moe/expert_balancing"

    @property
    def needs_router(self) -> bool:
        return bool(self.token_metrics or self.score_metrics or self.per_expert_tokens)

    @property
    def requested(self) -> bool:
        return bool(
            self.token_metrics
            or self.score_metrics
            or self.bias_metrics
            or self.per_expert_tokens
            or self.expert_weights
        )


@config(type="modelling")
class ModellingSourceConfig(FilteredSourceConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.callbacks.sources.modelling_source:ModellingSource"
    )

    # For a module y = f(x), the four capture flags select which of the
    # four tensors around it are logged. All of them are ACTIVATION
    # tensors — parameter gradients (dL/dW) belong to OptimizerSource.
    #
    #   forward:   x ──▶ [module] ──▶ y
    #   backward:  dL/dx ◀── [module] ◀── dL/dy

    # Module outputs y: activations_{stat}/{fqn}_output.{i}.
    capture_forward: bool = True
    # Output gradients dL/dy — what the module receives in backward:
    # activations_gradients_{stat}/...
    capture_backward: bool = False
    # Module inputs x: ..._input.{i}. Needed for tensors that are nobody's
    # output — the residual stream is only visible as a norm input (e.g.
    # ffn_norm input is the post-attention residual).
    capture_inputs: bool = False
    # Input gradients dL/dx — what the module passes down to earlier
    # layers: ..._grad_input.{i}.
    capture_grad_inputs: bool = False
    axes: list[str | int] = field(
        default_factory=lambda: ["batch", "sequence", "hidden"]
    )
    output: CaptureOutput = CaptureOutput.first_tensor
    metrics: list[MetricConfig] = field(default_factory=_default_tensor_metrics)
    moe: MoECaptureConfig = field(default_factory=MoECaptureConfig)
    # SwiGLU clamp counts of every FeedForward / GroupedExperts
    capture_swiglu_clips: bool = False
    swiglu_clip_namespace: str = "swiglu_clip"

    def __post_init__(self) -> None:
        missing = [
            metric.name
            for metric in self.metrics
            if not any(
                step.kind == MetricStepKind.reduce
                and step.reduce is not None
                and bool(step.reduce.group)
                and step.reduce.op != ReduceOp.none
                for step in metric.effective_steps()
            )
        ]
        if missing:
            raise ValueError(
                f"ModellingSource metrics {missing} declare no reduce "
                "stages, so there is no operator to combine the captures "
                "of one batch with. A module runs once per microbatch: "
                "with gradient accumulation this fails at the first log "
                "step. Give each metric a reduce op."
            )

    @property
    def uses_module_hooks(self) -> bool:
        return bool(
            self.capture_forward
            or self.capture_inputs
            or self.capture_backward
            or self.capture_grad_inputs
            or self.capture_swiglu_clips
            or self.moe.needs_router
        )
