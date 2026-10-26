import math
from collections.abc import Mapping
from dataclasses import replace

import torch
from torch import nn
from torch.utils._python_dispatch import _disable_current_modes

from typing import TypeAlias

from torchtitan.config.schema.callbacks.sources.optimizer_source import (
    OptimizerSourceConfig,
)
from torchtitan.callbacks.base import ObserverState
from torchtitan.callbacks.metrics.reducers import ParallelReducer
from torchtitan.callbacks.metrics.sample import Sample, MetricRecord, MetricValue
from torchtitan.config.schema.callbacks.metrics.sample import (
    AxisSpec,
    DTensorMode,
    MetricConfig,
    StatOpConfig,
    MetricStepKind,
    ReduceStageConfig,
)
from torchtitan.callbacks.metrics.stats import (
    _ACC_DTYPE,
    _assert_order_stat_size,
    _local_tensor,
    _validate_reduce_mode,
    _warn_full_tensor_fallback,
    apply_stat_op,
    MetricPipeline,
)
from torchtitan.callbacks.metrics.timers import callback_timer
from torchtitan.callbacks.sources.base import (
    MonitorSource,
    canonical_fqn,
)
from torchtitan.callbacks.sources.modelling_source import _segment_expert_stat
from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor

# Optimizer tensors are 1-axis samples over the flattened parameter. Frozen
# configs, so the hot per-parameter paths share these instead of rebuilding
# them every step.
_PARAM_AXES = AxisSpec(names=("param",))
# Shared read-only: these lists are aliased by every MetricConfig built below.
_NORM_PIPELINE = [StatOpConfig(op="norm")]
_ABS_MAX_PIPELINE = [StatOpConfig(op="abs_max")]
_NORM_METRIC = MetricConfig(name="norm", pipeline=_NORM_PIPELINE)


def _state_step_to_float(value: object) -> float | None:
    if hasattr(value, "to_local"):
        value = value.to_local()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


@torch.no_grad()
@_disable_current_modes()
def compute_segment_metric(
    value: MetricValue,
    metric: MetricConfig,
    reducer: ParallelReducer,
    *,
    num_segments: int,
    num_groups: int,
) -> tuple[torch.Tensor, tuple[ReduceStageConfig, ...]]:
    """Compute per-segment statistics, leaving trailing reductions pending."""
    _validate_reduce_mode(metric)
    steps = metric.effective_steps()
    if not steps:
        # Ratios already have one value per segment; scalar hyperparameters
        # (weight decay) apply to all segments.
        return torch.as_tensor(value).expand(num_segments), ()
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Segment metric '{metric.name}': expected a tensor")
    if num_segments <= 0 or num_groups <= 0 or num_segments % num_groups:
        raise ValueError("Segments must evenly divide the logical weight groups")
    if value.numel() == 0 or value.numel() % num_segments:
        raise ValueError("Segments must evenly divide a nonempty parameter")
    if metric.dtensor_mode == DTensorMode.local:
        raise ValueError(f"Segment metric '{metric.name}': local mode is unsupported")

    stats: list[StatOpConfig] = []
    stages: list[ReduceStageConfig] = []
    for step in steps:
        if step.kind == MetricStepKind.reduce:
            assert step.reduce is not None
            stages.append(step.reduce)
        else:
            assert step.stat is not None
            if stages or step.stat.dim is not None:
                raise ValueError(
                    f"Segment metric '{metric.name}': expected dimension-free stats before reductions"
                )
            stats.append(step.stat)
    if not stats or stats[-1].op in ("abs", "bincount"):
        raise ValueError(
            f"Segment metric '{metric.name}': expected a terminal scalar stat"
        )
    if any(op.op != "abs" for op in stats[:-1]):
        raise ValueError(
            f"Segment metric '{metric.name}': leading stats must be elementwise"
        )
    # Validate while the original shard placements are still available.
    MetricPipeline(metric, reducer)._guard_double_reduction(value, tuple(stages))
    _assert_order_stat_size(stats[-1].op, value.numel(), metric.name)
    if isinstance(value, Float8BlockwiseQTensor):
        value = _local_tensor(value)
    vector = value.detach().float()
    for op in stats[:-1]:
        vector = apply_stat_op(vector, AxisSpec(), op)
    op = stats[-1]
    support = reducer.shard_stats
    ragged = isinstance(vector, reducer.ragged_tensor_types)
    if ragged and metric.dtensor_mode == DTensorMode.partial_stats:
        assert support is not None
        try:
            result, finish = _segment_expert_stat(
                support,
                vector,
                op,
                metric.name,
                num_segments=num_segments,
                num_groups=num_groups,
            )
        except NotImplementedError:
            _warn_full_tensor_fallback(metric, op, vector)
        else:
            result = _local_tensor(result, full_tensor=True, dtype=_ACC_DTYPE)
            if finish is not None:
                result = finish(result)
            return result.float(), tuple(stages)
    if ragged or metric.dtensor_mode == DTensorMode.full_tensor:
        vector = _local_tensor(vector, full_tensor=True, shard_stats=support)
    rows = vector.reshape(num_segments, -1)
    op = replace(op, dim=1)
    if op.op == "actrel":
        # Each logical weight has its own threshold, shared across its experts.
        groups = rows.reshape(-1, num_groups, rows.shape[1]).unbind(1)
        result = torch.stack(
            [apply_stat_op(group, AxisSpec(), op) for group in groups], dim=1
        ).flatten()
    else:
        result = apply_stat_op(rows, AxisSpec(), op)
    return _local_tensor(result, full_tensor=True, dtype=torch.float32), tuple(stages)


class OptimizerSource(MonitorSource):
    """Logs optimizer, parameter, and gradient statistics.

    Gradient statistics are captured right before the optimizer step, i.e.
    before gradient clipping, consistent with the built-in
    ``l2_norm/grad/global``. Parameter and optimizer-state statistics are
    captured right after the optimizer step. All source-oriented tags are
    parameter-scoped (``optimizer/{grads,params,weight_decay}/{fqn}``,
    ``optimizer/states/{fqn}/{state}``): a parameter lives in exactly one
    optimizer group, and per-tensor lr/wd/clipping make these values
    parameter-level anyway. By default packed expert weights remain whole;
    per_expert_metrics reports each w1/w3 expert segment separately.

    Config:
        - ``gradients``: per-parameter gradient tags (both layouts)
        - ``parameters``: per-parameter weight tags (both layouts)
        - ``optimizer_states``: optimizer state, moment, and update tags
          (both layouts)
        - ``per_expert_metrics``: split packed expert tensors into logical
          w1/w3 segments for every expert
        - ``metrics``: what the source-oriented layout computes, as a
          declarative ``MetricConfig`` list (see the callbacks README). The
          statistic-oriented ``l2_norm/...``/``amax/...`` layout is fixed.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = OptimizerSourceConfig

    config: Config

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._named_params: list[tuple[str, nn.Parameter]] = []
        self._param_names_by_id: dict[int, str] = {}
        self._monitored_param_ids: set[int] = set()
        self._segment_names: dict[str, tuple[str, ...]] = {}
        self._group_hparams: dict[tuple[int, int], dict] = {}

    def fit_start(self, state: ObserverState) -> None:
        self._named_params.clear()
        self._param_names_by_id.clear()
        self._monitored_param_ids.clear()
        self._segment_names.clear()

        candidates: list[tuple[str, torch.Tensor]] = []
        for model_part in state.model_parts:
            if self.config.per_expert_metrics:
                declare = getattr(model_part, "named_weight_slices", None)
                slices = declare() if declare is not None else {}
            else:
                slices = {}
            for name, param in model_part.named_parameters():
                fqn = canonical_fqn(name)
                if not fqn:
                    continue
                self._param_names_by_id[id(param)] = fqn
                candidates.append((fqn, param))
                parts = slices.get(name)
                if parts is None or param.ndim != 4:
                    continue
                if param.shape[1] != 2:
                    continue
                expected_slices = [
                    (slice(None), slice(0, 1)),
                    (slice(None), slice(1, 2)),
                ]
                if [index for _, index in parts] != expected_slices:
                    continue
                self._segment_names[fqn] = tuple(
                    f"{canonical_fqn(logical_name)}__expert_idx_{expert}"
                    for expert in range(param.shape[0])
                    for logical_name, _ in parts
                )
        kept = self._filter_fqns([fqn for fqn, _ in candidates], "parameters")
        for fqn, param in candidates:
            if fqn in kept:
                self._named_params.append((fqn, param))
                self._monitored_param_ids.add(id(param))

    def before_optimizer_step(self, state: ObserverState) -> None:
        if not self.active:
            return
        # Pre-step hyperparameters, keyed by (optimizer part, group index):
        # the post-step statistics are read against the values the step used.
        self._group_hparams = {
            (group.part_index, group.group_index): {
                "lr": group.hparams["lr"],
                "weight_decay": group.hparams.get("weight_decay", 0.0),
                "betas": group.hparams.get("betas"),
                "eps": group.hparams.get("eps", 0.0),
            }
            for group in state.optimizer_groups()
        }
        if not self.config.gradients:
            return
        for name, param in self._named_params:
            if param.grad is None:
                continue
            self._record_gradient_metrics(param_name=name, grad=param.grad)

    @callback_timer(timer_level=2)
    def _record_gradient_metrics(self, *, param_name: str, grad: torch.Tensor) -> None:
        self._record(
            Sample(
                name=f"optimizer/grads/{param_name}",
                value=grad,
                axes=_PARAM_AXES,
                metrics=self.config.metrics,
            ),
            param_name=param_name,
        )
        self._record(
            Sample(
                name="l2_norm/grad",
                value=grad,
                axes=_PARAM_AXES,
                metrics=[MetricConfig(name=param_name, pipeline=_NORM_PIPELINE)],
            ),
            param_name=param_name,
        )

    def after_optimizer_step(self, state: ObserverState) -> None:
        if not self.active:
            return
        for group in state.optimizer_groups():
            hparams = self._group_hparams[(group.part_index, group.group_index)]
            for param in group.params:
                param_name = self._param_names_by_id.get(id(param))
                if param_name is None or id(param) not in self._monitored_param_ids:
                    continue

                state_dict = state.optimizer_state(param)
                if self.config.parameters:
                    self._record_parameter_metrics(param_name=param_name, param=param)
                if self.config.optimizer_states:
                    self._record_state_metrics(
                        param_name=param_name,
                        group=hparams,
                        state_dict=state_dict,
                    )
                    self._record_moment_metrics(
                        optimizer_type=group.optimizer_type,
                        param_name=param_name,
                        param=param,
                        state_dict=state_dict,
                        group=hparams,
                    )

    @callback_timer(timer_level=2)
    def _record_parameter_metrics(
        self, *, param_name: str, param: torch.Tensor
    ) -> None:
        self._record(
            Sample(
                name=f"optimizer/params/{param_name}",
                value=param,
                axes=_PARAM_AXES,
                metrics=self.config.metrics,
            ),
            param_name=param_name,
        )
        self._record(
            Sample(
                name="l2_norm/param",
                value=param,
                axes=_PARAM_AXES,
                metrics=[MetricConfig(name=param_name, pipeline=_NORM_PIPELINE)],
            ),
            param_name=param_name,
        )

    @callback_timer(timer_level=2)
    def _record_state_metrics(
        self, *, param_name: str, group: dict, state_dict: dict
    ) -> None:
        self._record(
            Sample(
                name="optimizer/weight_decay",
                value=float(group.get("weight_decay", 0.0)),
                metrics=[MetricConfig(name=param_name)],
            ),
            param_name=param_name,
        )
        for state_name, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                self._record(
                    Sample(
                        name=f"optimizer/states/{param_name}/{state_name}",
                        value=value,
                        axes=_PARAM_AXES,
                        metrics=self.config.metrics,
                    ),
                    param_name=param_name,
                )

    @callback_timer(timer_level=2)
    def _record_moment_metrics(
        self,
        *,
        optimizer_type: type,
        param_name: str,
        param: torch.Tensor,
        state_dict: Mapping[str, object],
        group: dict[str, object],
    ) -> None:
        norm_metric = [MetricConfig(name=param_name, pipeline=_NORM_PIPELINE)]
        abs_max_metric = [MetricConfig(name=param_name, pipeline=_ABS_MAX_PIPELINE)]

        exp_avg = state_dict.get("exp_avg")
        exp_avg_sq = state_dict.get("exp_avg_sq")
        for state_name, tensor in (("moment", exp_avg), ("moment_sq", exp_avg_sq)):
            if not isinstance(tensor, torch.Tensor):
                continue
            self._record(
                Sample(
                    name=f"l2_norm/{state_name}",
                    value=tensor,
                    axes=_PARAM_AXES,
                    metrics=norm_metric,
                ),
                param_name=param_name,
            )
            self._record(
                Sample(
                    name=f"amax/{state_name}",
                    value=tensor,
                    axes=_PARAM_AXES,
                    metrics=abs_max_metric,
                ),
                param_name=param_name,
            )

        step = _state_step_to_float(state_dict.get("step"))
        betas = group.get("betas")
        if (
            step is None
            or step <= 0
            or not isinstance(betas, tuple)
            or len(betas) != 2
            or not isinstance(exp_avg, torch.Tensor)
            or not isinstance(exp_avg_sq, torch.Tensor)
        ):
            return

        beta1, beta2 = float(betas[0]), float(betas[1])
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        for name, tensor in (
            ("l2_norm/moment_bc", exp_avg / bias_correction1),
            ("l2_norm/moment_sq_bc", exp_avg_sq / bias_correction2),
        ):
            self._record(
                Sample(
                    name=name,
                    value=tensor,
                    axes=_PARAM_AXES,
                    metrics=norm_metric,
                ),
                param_name=param_name,
            )

        update = self._update_tensor(
            optimizer_type=optimizer_type,
            param=param,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            group=group,
            bias_correction1=bias_correction1,
            bias_correction2=bias_correction2,
        )
        if update is not None:
            self._record(
                Sample(
                    name="l2_norm/update",
                    value=update,
                    axes=_PARAM_AXES,
                    metrics=norm_metric,
                ),
                param_name=param_name,
            )
        self._record_update_ratios(param_name=param_name, param=param, update=update)

    def _record_update_ratios(
        self,
        *,
        param_name: str,
        param: torch.Tensor,
        update: torch.Tensor | None,
    ) -> None:
        """Derived per-parameter ratios of the update, gradient, and weight norms.

        Measured after the optimizer step: the gradient is post-clipping (the
        published l2_norm/grad/{fqn} is pre-clipping, so these ratios are not
        recomputable from the published curves when clipping engages).
        """
        eps = 1e-8
        param_norm = self._norm_value(param, param_name=param_name)
        grad_norm = (
            self._norm_value(param.grad, param_name=param_name)
            if param.grad is not None
            else None
        )
        update_norm = (
            self._norm_value(update, param_name=param_name)
            if update is not None
            else None
        )
        ratios: dict[str, torch.Tensor] = {}
        if update_norm is not None and grad_norm is not None:
            ratios["effective_lr"] = update_norm / (grad_norm + eps)
        if update_norm is not None:
            ratios["update_weight_ratio"] = update_norm / (param_norm + eps)
        if grad_norm is not None:
            ratios["update_rate"] = grad_norm / (param_norm + eps)
        for name, value in ratios.items():
            self._record(
                Sample(
                    name=name,
                    value=value,
                    metrics=[MetricConfig(name=param_name)],
                ),
                param_name=param_name,
            )

    def _norm_value(self, tensor: torch.Tensor, *, param_name: str) -> torch.Tensor:
        assert self.reducer is not None, "MonitorSource records before fit_start"
        names = self._segment_names.get(param_name)
        if names is not None:
            return compute_segment_metric(
                tensor,
                _NORM_METRIC,
                self.reducer,
                num_segments=len(names),
                num_groups=2,
            )[0]
        return MetricPipeline(_NORM_METRIC, self.reducer).compute(tensor, _PARAM_AXES)

    @torch.no_grad()
    def _record(
        self, sample: Sample, *, param_name: str | None = None, accumulate: bool = False
    ) -> None:
        if param_name is None or param_name not in self._segment_names:
            super()._record(sample, accumulate=accumulate)
            return
        names = self._segment_names[param_name]
        if not self.active or torch.compiler.is_compiling() or not sample.metrics:
            return
        if accumulate:
            raise ValueError("Packed optimizer metrics are recorded once per step")
        assert self.reducer is not None, "MonitorSource records before fit_start"
        for metric in sample.metrics:
            variants = [("", metric)]
            if (
                self.debug_full_tensor
                and isinstance(sample.value, self.reducer.sharded_tensor_types)
                and metric.dtensor_mode == DTensorMode.partial_stats
            ):
                variants.append(
                    (
                        "debug_full/",
                        replace(metric, dtensor_mode=DTensorMode.full_tensor),
                    )
                )
            for prefix, variant in variants:
                values, stages = compute_segment_metric(
                    sample.value,
                    variant,
                    self.reducer,
                    num_segments=len(names),
                    num_groups=2,
                )
                tag = prefix + self._resolve_tag(sample.name, metric.name)
                for name, value in zip(names, values):
                    self._records.append(
                        MetricRecord(
                            name=tag.replace(param_name, name),
                            value=value,
                            step=self._step,
                            reduce=variant.reduce,
                            pending_stages=stages,
                        )
                    )

    def _update_tensor(
        self,
        *,
        optimizer_type: type,
        param: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        group: dict[str, object],
        bias_correction1: float,
        bias_correction2: float,
    ) -> torch.Tensor | None:
        """Exactly reconstruct the last parameter update.

        torch.optim.AdamW steps as ``p_post = p_pre * (1 - lr*wd) - A`` with
        the adam term ``A = lr * m_hat / (sqrt(v_hat) + eps)`` (the moments
        are updated before the step, so the post-step state is exactly what
        the step used). This runs after the step, so the pre-step parameter
        is recovered algebraically and the update collapses to
        ``(A + lr*wd*p_post) / (1 - lr*wd)``. torch.optim.Adam folds L2 decay
        into the gradient (already inside the moments), so its update is
        ``A``. Other optimizers are not reconstructed.
        """
        lr = float(group["lr"])
        if issubclass(optimizer_type, torch.optim.AdamW):
            weight_decay_factor = lr * float(group.get("weight_decay", 0.0))
        elif issubclass(optimizer_type, torch.optim.Adam):
            weight_decay_factor = 0.0
        else:
            raise ValueError(
                f"OptimizerSource cannot reconstruct updates for "
                f"{optimizer_type.__name__}: only Adam/AdamW are implemented. "
                "Extend _update_tensor when adding a new optimizer."
            )

        eps = float(group.get("eps", 0.0))
        denom = exp_avg_sq.float().sqrt() / math.sqrt(bias_correction2) + eps
        adam_term = (lr / bias_correction1) * exp_avg.float() / denom
        if weight_decay_factor == 0.0:
            return adam_term
        return (adam_term + weight_decay_factor * param.detach().float()) / (
            1.0 - weight_decay_factor
        )
