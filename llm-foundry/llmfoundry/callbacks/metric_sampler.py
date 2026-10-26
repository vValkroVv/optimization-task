from __future__ import annotations

import functools
import logging
import math
import os
import re
import socket
import time
from pathlib import Path
from typing import Callable, Protocol

import torch
from composer.algorithms import GradientClipping
from composer.core import Callback, Precision, State
from composer.loggers import Logger
from composer.utils import dist, global_actlog_state

from llmfoundry.callbacks.metric_sampler_monitors import RankLocalMonitorCapture
from llmfoundry.callbacks.metric_sampler_weights import MaterializedWeights
from llmfoundry.callbacks.expert_balancing_monitor import get_moe_gates
from llmfoundry.callbacks.token_dispatcher_monitor import get_moe_token_dispatchers
from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM
from llmfoundry.models.layers.attention import LlamaLatentAttention, Qwen3NextGatedDeltaNet
from llmfoundry.models.layers.attn_row_stats import attention_row_stats
from llmfoundry.models.layers.moe import AbstractGMMMoeBlock, ScMoEBlock
from llmfoundry.models.layers.moe.dispatchers import TokenDispatcher
from llmfoundry.models.layers.moe.experts import GroupedLlamaMLP
from llmfoundry.models.layers.moe.routers import MoEGate
from metric_sampler.capture.dump import ActivationDumpConfigError, DumpCapture, DumpSettings
from metric_sampler.capture.kernels import activation_ext, token_loss_kernel, triton_launch
from metric_sampler.capture.layout import Layout
from metric_sampler.capture.layout_plan import (
    DISPATCHER_METRICS,
    LayoutPlan,
    LinearAttentionLayer,
    SoftmaxAttentionLayer,
)
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder, ResolvedSlot
from metric_sampler.capture.stats.activations import ActivationStats
from metric_sampler.capture.stats.attention import AttentionStats
from metric_sampler.capture.stats.helpers.quantiles import exact_quantiles, row_entropy
from metric_sampler.capture.stats.monitor_activations import MonitorActivationStats, MonitorObservations
from metric_sampler.capture.stats.quantiles import ExactQuantileStats
from metric_sampler.capture.stats.token_loss import TokenLossStats
from metric_sampler.contract.metrics import LAYER_ACTIVATION_POINTS, NO_LAYER, NO_MICROBATCH, MetricSpec
from metric_sampler.rank_capture import RankCapture, RankPlacement

__all__ = ["MetricSampler", "attempt_id"]

log = logging.getLogger(__name__)

LOSS_METRICS = {
    "total": "loss/total",
    "loss_ce": "loss/ce",
    "loss_mtp": "loss/mtp",
    "z_loss": "loss/z_loss",
    "hidden_z_loss_attn": "loss/hidden_z_loss_attn",
    "hidden_z_loss_moe": "loss/hidden_z_loss_moe",
    "hidden_z_loss_dense_mlp": "loss/hidden_z_loss_dense_mlp",
    "hidden_z_loss_mla_kv_latent": "loss/hidden_z_loss_mla_kv_latent",
    "hidden_z_loss_mtp0_mla_kv_latent": "loss/hidden_z_loss_mtp0_mla_kv_latent",
}

# SuperTensor names are dotted module paths rooted at GigaMixModel.
_MAIN_LAYER_NAME_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")
_MTP_LAYER_NAME_PATTERN = re.compile(r"^model\.mtp_block\.mtp_layers\.(\d+)\.")

# use_monitor_variable names a main decoder loop point with this template, its layer given apart, and an MTP point
# with the predictor index in the name (layer_depends=False). Without the layer part both forms give the base
# name that keys _ACTIVATION_POINTS.
_MAIN_LAYER_ACTLOG_PREFIX = "model.model.layers.{}"
_MTP_ACTLOG_NAME_PATTERN = re.compile(r"^model\.model\.mtp_block\.(\d+)(?:\.(.*))?$")

# (base name, suffix) -> ActivationStats point. A pair that is not listed is not an activation point.
_ACTIVATION_POINTS: dict[tuple[str, str], str] = {
    ("", "_output.0"): "residual",
    ("self_attn._attn", "_output.0"): "attn_out",
    ("self_attn.mlp", "_output.0"): "mlp_out",
    ("block_sparse_moe", "_output.0"): "mlp_out",
    ("self_attn.shared_experts", "_output.0"): "shared_expert_out",
    ("attn_residual_contribution", "_output.0"): "attn_residual_contribution",
    ("mlp_residual_contribution", "_output.0"): "mlp_residual_contribution",
    ("block_sparse_moe.fused_swiglu", "_input.0"): "swiglu_in",
    ("block_sparse_moe.gmm_down", "_input.0"): "down_in",
}

# Points outside the decoder layers, recorded without a layer.
_STANDALONE_ACTIVATION_POINTS: dict[tuple[str, str], str] = {
    ("model.norm", "_output.0"): "final_norm",
    ("model.model.embed_tokens", "_output.0"): "embed",
}

# A (name, suffix) pair not resolved yet; None marks a resolved pair that is not an activation point.
_UNRESOLVED = object()

# torch's graph task id outside a backward pass: a call made inside one recomputes a checkpointed forward.
_OUTSIDE_BACKWARD = -1
# The GigaFSDP zero stage whose gradient reduction the pre-reduce statistics observe.
_PRE_REDUCE_ZERO_STAGE = 1
# Router scores are a [tokens, experts] matrix.
_ROUTER_SCORES_NDIM = 2
_BYTES_PER_GB = 1e9

# The decoder layers of the model with their layer index, the main layers and then those of the MTP heads.
_DecoderLayers = list[tuple[int, torch.nn.Module]]
# A SuperTensor with the metric its gradient norm is recorded as, its layer and its communication stream.
_PreReduceTargets = list[tuple[object, str, int, torch.cuda.Stream]]


class ActlogActivationObserver:
    """Bridges `global_actlog_state.use_monitor_variable` calls into `ActivationStats.observe`.

    Runs on every actlog call site regardless of `enable_monitor`."""

    def __init__(
        self,
        stats: ActivationStats,
        num_layers: int,
        dump_observe: Callable[[str, int, torch.Tensor], None] | None = None,
    ) -> None:
        self._stats = stats
        self._num_layers = num_layers
        self._dump_observe = dump_observe
        # Value is (point, layer): layer is None for a main decoder loop point, whose layer index comes with each
        # call, or fixed by the name for an MTP or a standalone point.
        self._resolved: dict[tuple[str, str], tuple[str, int | None] | None] = {}

    def __call__(
        self,
        tensor: torch.Tensor,
        name: str,
        suffix: str,
        layer_depends: bool,
        layer_index: int,
    ) -> None:
        key = (name, suffix)
        resolved = self._resolved.get(key, _UNRESOLVED)
        if resolved is _UNRESOLVED:
            resolved = self._resolved[key] = self.resolve(name, suffix)
        if resolved is None:
            return
        point, static_layer = resolved
        layer = layer_index if static_layer is None else static_layer
        self._stats.observe(point, layer, tensor, training_forward=True)
        if self._dump_observe is not None:
            self._dump_observe(point, layer, tensor)

    def resolve(self, name: str, suffix: str) -> tuple[str, int | None] | None:
        """The point and fixed layer of an actlog call (None: the call's layer index), or None for a call that
        is not an activation point."""
        # Main decoder loop: `name` is the unresolved template, the layer comes from the call's layer index.
        if name == _MAIN_LAYER_ACTLOG_PREFIX:
            base = ""
        elif name.startswith(_MAIN_LAYER_ACTLOG_PREFIX + "."):
            base = name[len(_MAIN_LAYER_ACTLOG_PREFIX) + 1 :]
        else:
            base = None
        if base is not None:
            point = _ACTIVATION_POINTS.get((base, suffix))
            return None if point is None else (point, None)

        # MTP head: `name` already has the predictor index baked in (`layer_depends=False`).
        mtp_match = _MTP_ACTLOG_NAME_PATTERN.match(name)
        if mtp_match is not None:
            predictor_index, base = mtp_match.group(1), mtp_match.group(2) or ""
            point = _ACTIVATION_POINTS.get((base, suffix))
            return None if point is None else (point, self._num_layers + int(predictor_index))

        point = _STANDALONE_ACTIVATION_POINTS.get((name, suffix))
        return None if point is None else (point, NO_LAYER)


class _NativeRouting(Protocol):
    """The extension's RoutingCalls: the routed layers' routing observers, made from C++."""

    def router(self, layer: int) -> Callable[..., None]: ...


class _GuardedObserver:
    """Calls `observer` from the model's forward and backward on behalf of the sampler. A capture capacity failure is
    raised and stops training. Any other failure disables the sampler instead of raising through the model, unless
    the sampler's failures stop training. `where` names the observer in the log line of the failure."""

    def __init__(self, observer: Callable[..., None], sampler: "MetricSampler", where: str) -> None:
        self._observer = observer
        self._sampler = sampler
        self._where = where

    def __call__(self, *args, **kwargs) -> None:
        try:
            self._observer(*args, **kwargs)
        except CaptureCapacityError:
            raise
        except Exception as error:
            if self._sampler.failures_stop_training:
                raise
            self._sampler.disable(error, self._where)


def attempt_id() -> str:
    """Identifies one training attempt, shared by the sampler and its logger destination."""
    return f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.environ.get('SLURM_RESTART_COUNT', '0')}"


class PreReduceGradObserver:
    """Accumulates one GigaFSDP SuperTensor's local gradient norm before its reduce-scatter.

    Called from `SuperTensor.reduce_scatter_grad` on the autograd engine, but only while the main
    thread is blocked inside `backward()` for the step's last microbatch, so there's no race."""

    def __init__(self, recorder: RankRecorder, metric_name: str, layer: int, comm_stream: torch.cuda.Stream) -> None:
        self._recorder = recorder
        self._metric_name = metric_name
        self._layer = layer
        self._comm_stream = comm_stream
        self.event: torch.cuda.Event | None = None

    def __call__(self, grad: torch.Tensor) -> None:
        self._recorder.accumulator(self._metric_name, self._layer, NO_MICROBATCH).add_(
            torch.linalg.vector_norm(grad, dtype=torch.float32).square()
        )
        self.event = torch.cuda.Event()
        self.event.record(self._comm_stream)


class MetricSampler(Callback):
    """Records raw per-rank, per-microbatch MoE telemetry every step.

    Args:
        output_dir (str): Root directory for node windows (`attempt=<id>/node=<host>/w<a>-<b>/`).
        window_steps (int): Number of steps per written window. Defaults to 10.
        distribution_interval (int): Step interval for detailed-mode capture (per-expert and
            per-token-loss histograms); also triggered by the ring's own verbose
            window. Defaults to 100.
        activation_dump (dict | None): Raw activation dump config, or None (the default) to
            disable dumps entirely. Keys: `interval` (steps; 0 or missing disables dumps),
            `points` (list of hidden-width activation point names, required when interval > 0),
            `layers` (list of decoder layer indices, required when interval > 0), `tokens`
            (first N rows of microbatch 0's x2d per point; defaults to 256), `max_gb` (per-attempt
            per-rank byte budget; defaults to 10.0). Validated on fit_start; a bad config raises
            `ActivationDumpConfigError` there instead of disabling the sampler.
        attention_stats (bool): Records exact per-layer, per-microbatch statistics of softmax (MLA)
            and linear (GDN) attention every step; a failure then stops training instead of
            disabling the sampler. Defaults to False.
        capture_interval (int): Captures the steps whose number is a multiple of it; on the other
            steps the model's and the monitors' observers are detached, so they do no capture work.
            `distribution_interval` and an activation dump `interval` must be multiples of it.
            Defaults to 1, every step.
        activations (bool): Records the activation statistics of every layer and point (`act/*`);
            activation dumps need them. Defaults to True.
        token_loss (bool): Records the token-loss histograms, stream, position, sample and MTP
            statistics (`loss/*` beyond the loss components). Defaults to True.
        experts (bool): Records the routing, dispatcher and expert-loss statistics of the MoE layers
            (`router/*`, `dispatcher/*`, `expert_loss/*`). Defaults to True.
    """

    def __init__(
        self,
        output_dir: str,
        window_steps: int = 10,
        distribution_interval: int = 100,
        activation_dump: dict | None = None,
        exact_quantiles: bool = False,
        rank_local_monitors: bool = False,
        attention_stats: bool = False,
        capture_interval: int = 1,
        activations: bool = True,
        token_loss: bool = True,
        experts: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.window_steps = window_steps
        self.distribution_interval = distribution_interval
        self.capture_interval = capture_interval
        self.activations = activations
        self.token_loss = token_loss
        self.experts = experts
        self.activation_dump = activation_dump
        self.exact_quantiles = exact_quantiles
        self.rank_local_monitors = rank_local_monitors
        self.attention_stats_enabled = attention_stats
        self._init_step_state()
        self._init_capture_state()
        self.monitor_capture = RankLocalMonitorCapture(self)

    def _init_step_state(self) -> None:
        # Whether the step in progress is captured, and whether the observers the model and the monitors call on
        # their own are attached (None before fit_start attaches or detaches them for the first step).
        self._capturing = True
        self._step_observers_active: bool | None = None
        self._capture_forward_active = False
        self._grad_forward = False
        self._dump_budget_logged = False
        self.microbatches = 0
        self.microbatch = 0
        self.disabled = False
        self._first_batch_start = True
        self.step_started = 0.0
        self.last_step_end_s = 0.0
        self.unknown_loss_keys: set[str] = set()
        self._step_start_event: torch.cuda.Event | None = None
        self._forward_start_event: torch.cuda.Event | None = None
        self._backward_start_event: torch.cuda.Event | None = None
        self._optimizer_start_event: torch.cuda.Event | None = None

    def _init_capture_state(self) -> None:
        self.quantile_stats: ExactQuantileStats | None = None
        self._mtp_layers: list[int] = []
        self._hook_handles: list = []
        # Per (layer, microbatch): the slots of the routed tokens' entropy quantiles and their counts, and those of
        # the dispatcher's assignment counts.
        self._entropy_quantile_slots: dict[tuple[int, int], tuple[ResolvedSlot, ResolvedSlot]] = {}
        self._dispatcher_count_slots: dict[tuple[int, int], tuple[ResolvedSlot, ...]] = {}
        # The forward in progress' MonitorObservations, once its first monitored call made them: the extension's
        # module hooks hand their calls to it.
        self._monitor_observations: MonitorObservations | None = None
        self.counting_gates: list[tuple[int, MoEGate]] = []
        self.dispatchers: list[tuple[int, TokenDispatcher]] = []
        self._dispatcher_observers: list[tuple[TokenDispatcher, _GuardedObserver]] = []
        self.recorder: RankRecorder | None = None
        # The rank's end of capture, from fit_start; `recorder` is its recorder, held here for the observers.
        self.rank_capture: RankCapture | None = None
        self.causal_lm: GigaMixForCausalLM | None = None
        self.token_loss_stats: TokenLossStats | None = None
        self.activation_stats: ActivationStats | None = None
        # The guarded Python observer of actlog calls, or the extension's (see _install_activation_observer).
        self._activation_observer: object | None = None
        self.gradient_clipping: GradientClipping | None = None
        self.pre_reduce_observers: list[PreReduceGradObserver] = []
        self._routing_observers: list[tuple[MoEGate, _GuardedObserver]] = []
        self._loss_observer: _GuardedObserver | None = None
        self._pre_reduce_guards: list[tuple[object, _GuardedObserver]] = []
        self._routed_gates: list[tuple[int, MoEGate]] = []
        self._pre_reduce_super_tensors: list[object] = []
        self._expert_modules: list[tuple[int, GroupedLlamaMLP]] = []
        self.attention_stats: AttentionStats | None = None
        self._softmax_attention: list[tuple[int, LlamaLatentAttention]] = []
        self._linear_attention: list[tuple[int, Qwen3NextGatedDeltaNet]] = []
        self._attention_observers: list[tuple[torch.nn.Module, _GuardedObserver]] = []
        self._gate_observers: list[tuple[LlamaLatentAttention, _GuardedObserver]] = []

    @functools.cached_property
    def failures_stop_training(self) -> bool:
        """Whether a failed observer raises through the model, stopping training, instead of disabling the sampler:
        exact quantiles, rank-local monitors and attention statistics are data a silent disable would lose."""
        return self.exact_quantiles or self.rank_local_monitors or self.attention_stats_enabled

    @property
    def monitor_activations(self) -> MonitorActivationStats | None:
        return self.monitor_capture.monitor_activations

    @monitor_activations.setter
    def monitor_activations(self, stats: MonitorActivationStats | None) -> None:
        self.monitor_capture.monitor_activations = stats

    @property
    def materialized_weights(self) -> MaterializedWeights | None:
        return self.monitor_capture.materialized_weights

    @materialized_weights.setter
    def materialized_weights(self, weights: MaterializedWeights | None) -> None:
        self.monitor_capture.materialized_weights = weights

    def fit_start(self, state: State, logger: Logger):
        self._activation_dump_config = DumpSettings.parse(self.activation_dump)
        self._fit_start(state, logger)

    def _fit_start(self, state: State, logger: Logger):
        self._validate_settings(state)
        # GigaFSDP forwards attribute reads but not writes; observers belong on the underlying model.
        self.causal_lm = next(module for module in state.model.modules() if isinstance(module, GigaMixForCausalLM))
        if self.causal_lm.model.gradient_checkpointing:
            raise ValueError(
                "metric_sampler requires the trainer's activation_checkpointing; "
                "model.gradient_checkpointing uses an unsupported reentrant forward"
            )
        self._require_contiguous_node_ranks()
        attempt = attempt_id()
        self._routed_gates = []
        self._pre_reduce_super_tensors = []

        num_layers = len(self.causal_lm.model.layers)
        gates, gate_tags = get_moe_gates(state)
        dispatchers, dispatcher_tags = get_moe_token_dispatchers(state)
        # Only the plain MoEGate accumulates local_tokens_per_expert; Uniform, FirstExpertGroup and
        # Dynamic gates never do, so their layers have dispatcher counts only.
        self.counting_gates = [
            (self._layer_index(tag, num_layers), gate)
            for gate, tag in zip(gates, gate_tags)
            if self.experts and type(gate) is MoEGate and gate.local_tokens_per_expert is not None
        ]
        self.dispatchers = [
            (self._layer_index(tag, num_layers), dispatcher)
            for dispatcher, tag in zip(dispatchers, dispatcher_tags)
            if self.experts
        ]
        self.microbatches = math.ceil(state.train_dataloader.batch_size / state.device_train_microbatch_size)

        mtp_decoder_layers = self._mtp_decoder_layers(num_layers)
        mtp_layers = [layer for layer, _ in mtp_decoder_layers]
        activation_layers = self._activation_point_layers(mtp_decoder_layers)
        if self.attention_stats_enabled:
            self._find_attention_layers(mtp_decoder_layers)
        routed_layers = self._collect_routed_layers(gates, gate_tags, num_layers)

        experts = gates[0].n_routed_experts
        top_k = gates[0].top_k
        stream_names = tuple(getattr(state.train_dataloader.dataset, "stream_names", None) or ("all",))

        pre_reduce_targets = self._gigafsdp_pre_reduce_targets(state, num_layers)
        pre_reduce_keys = {(metric_name, layer) for _, metric_name, layer, _ in pre_reduce_targets}

        self._mtp_layers = mtp_layers
        if self.rank_local_monitors:
            self._find_expert_modules(mtp_decoder_layers)
            self.monitor_capture.prepare(state, self._routed_gates, self._expert_modules)

        dump_entries = (
            self._activation_dump_config.entries(activation_layers) if self._activation_dump_config.interval > 0 else ()
        )
        dumps_enabled = bool(dump_entries)

        layout = Layout(
            self._layout_requests(
                experts, routed_layers, mtp_layers, len(stream_names), pre_reduce_keys, activation_layers,
                dumps_enabled=dumps_enabled,
            ),
            stream_names=stream_names,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        self.rank_capture = RankCapture(self.output_dir, self._rank_placement(attempt), layout, device)
        self.recorder = self.rank_capture.recorder
        self._dispatcher_observers = [
            (dispatcher, self.layer_observer("assignment_observer", layer, self._observe_dispatcher))
            for layer, dispatcher in self.dispatchers
        ]
        if self.attention_stats_enabled:
            self.attention_stats = AttentionStats(self.recorder)
            self._build_attention_observers()

        if dumps_enabled:
            self.rank_capture.start_dumps(
                self._activation_dump_config.config(dump_entries), self.causal_lm.config.hidden_size
            )

        self._install_activation_observer(num_layers, dumps_enabled)
        self._install_routing_and_loss_observers(experts, top_k, routed_layers, stream_names, dumps_enabled)

        if self.rank_local_monitors:
            self.monitor_capture.install(state, attempt)

        self._install_pre_reduce_observers(pre_reduce_targets)
        self._set_step_observers(self._captures(state.timestamp.batch.value + 1))

        self.gradient_clipping = next(
            (algorithm for algorithm in state.algorithms if isinstance(algorithm, GradientClipping)), None
        )
        self.rank_capture.start_node(
            self.window_steps, self.capture_interval, dumps_enabled=self._activation_dump_config.interval > 0
        )

    def _validate_settings(self, state: State) -> None:
        if (dist.get_tp_group_size() or 1) != 1 or (dist.get_sp_group_size() or 1) != 1:
            raise ValueError("metric_sampler requires tensor and sequence parallel sizes == 1")
        if self.distribution_interval <= 0 or self.window_steps <= 0 or self.capture_interval <= 0:
            raise ValueError("metric_sampler window_steps, distribution_interval and capture_interval must be positive")
        if self.distribution_interval % self.capture_interval:
            raise ValueError("metric_sampler distribution_interval must be a multiple of capture_interval: "
                             "detailed steps are captured steps")
        dump_interval = self._activation_dump_config.interval
        if dump_interval > 0 and (dump_interval % self.capture_interval or not self.activations):
            raise ActivationDumpConfigError(
                "activation_dump interval must be a multiple of capture_interval, and dumps need activations")
        if state.auto_microbatching:
            raise ValueError(
                "metric_sampler requires a fixed device_train_microbatch_size; auto_microbatching is unsupported"
            )
        if state.precision == Precision.AMP_FP16:
            raise ValueError("metric_sampler pre-reduce gradient statistics do not support AMP FP16 gradient scaling")

    @staticmethod
    def _require_contiguous_node_ranks() -> None:
        assert dist.get_global_rank() == dist.get_node_rank() * dist.get_local_world_size() + dist.get_local_rank(), (
            "metric_sampler assumes contiguous global ranks per node"
        )

    @staticmethod
    def _rank_placement(attempt: str) -> RankPlacement:
        return RankPlacement(
            attempt=attempt, node=socket.gethostname(), global_rank=dist.get_global_rank(),
            local_rank=dist.get_local_rank(), local_world_size=dist.get_local_world_size(),
            node_rank=dist.get_node_rank(), world_size=dist.get_world_size(),
        )

    def _mtp_decoder_layers(self, num_layers: int) -> _DecoderLayers:
        """The decoder layer of each MTP head the run uses, numbered after the main layers."""
        config = self.causal_lm.config
        if not config.use_mtp:
            return []

        mtp_layers_in_tree = getattr(getattr(self.causal_lm.model, "mtp_block", None), "mtp_layers", []) or []
        assert len(mtp_layers_in_tree) == config.mtp_predictor_num, (
            f"metric_sampler expected {config.mtp_predictor_num} MTP layers (config.mtp_predictor_num), "
            f"found {len(mtp_layers_in_tree)} in the tree"
        )
        return [(num_layers + i, mtp_layers_in_tree[i].decoder_layer) for i in range(config.mtp_predictor_num)]

    def _decoder_layers(self, mtp_decoder_layers: _DecoderLayers) -> _DecoderLayers:
        return list(enumerate(self.causal_lm.model.layers)) + mtp_decoder_layers

    def _collect_routed_layers(self, gates: list[MoEGate], gate_tags: list[str], num_layers: int) -> dict[int, int]:
        """Records in `_routed_gates` every plain MoEGate whose segment _compute_loss produces this run, and returns
        the loss segment of each routed layer (0 = main, head h = MTP layer h). MTP gates are left out when
        use_mtp is off, since _compute_loss then produces no MTP segments."""
        routed_layers: dict[int, int] = {}
        for gate, tag in zip(gates, gate_tags):
            if type(gate) is not MoEGate:
                continue
            kind, index = tag.rsplit("_", 1)
            if kind != "layer" and not self.causal_lm.config.use_mtp:
                continue
            layer = self._layer_index(tag, num_layers)
            segment = 0 if kind == "layer" else int(index) + 1
            self._routed_gates.append((layer, gate))
            routed_layers[layer] = segment
        return routed_layers

    def _find_expert_modules(self, mtp_decoder_layers: _DecoderLayers) -> None:
        for layer, decoder_layer in self._decoder_layers(mtp_decoder_layers):
            expert_module = self._experts_module(decoder_layer)
            if expert_module is not None:
                if not isinstance(expert_module, GroupedLlamaMLP):
                    raise ValueError("rank-local expert moments require GroupedLlamaMLP")
                self._expert_modules.append((layer, expert_module))

    def layer_observer(
        self, name: str, layer: int, observe: Callable[..., None], *arguments: object
    ) -> _GuardedObserver:
        """The guarded observer of one layer's `name` calls: `observe` with the layer and `arguments` ahead of the
        call's own."""
        return _GuardedObserver(functools.partial(observe, layer, *arguments), self, f"{name}(layer={layer})")

    def _install_activation_observer(self, num_layers: int, dumps_enabled: bool) -> None:
        if self.exact_quantiles:
            self.quantile_stats = ExactQuantileStats(self.recorder)
        # Activation quantiles are selected with the activation statistics: one read counts both.
        self.activation_stats = ActivationStats(self.recorder, self.quantile_stats)
        actlog_observer = ActlogActivationObserver(
            self.activation_stats, num_layers, self._dump_observe if dumps_enabled else None)
        # Without dumps, the extension takes an actlog call with exact quantiles, without a Python call; the guard
        # would re-raise each of its failures, as it does every failure with exact quantiles.
        native = None if dumps_enabled else self.activation_stats.native_observer(actlog_observer.resolve)
        if self.activations:
            self._activation_observer = native if native is not None else _GuardedObserver(
                actlog_observer, self, "activation_observer")
        self._hook_handles.append(self.causal_lm.register_forward_pre_hook(self._begin_model_pass))

    def _install_routing_and_loss_observers(
        self, experts: int, top_k: int, routed_layers: dict[int, int], stream_names: tuple[str, ...],
        dumps_enabled: bool,
    ) -> None:
        self.token_loss_stats = TokenLossStats(
            self.recorder, routed_layers if self.experts else {}, self._mtp_layers, experts, top_k, len(stream_names),
            token_loss=self.token_loss,
        )
        if self.experts:
            native_routing = self._native_routing(experts)
            if native_routing is not None:  # pragma: no cover
                self._routing_observers = [(gate, native_routing.router(layer)) for layer, gate in self._routed_gates]
            else:
                self._routing_observers = [
                    (gate, self.layer_observer("routing_observer", layer, self._observe_routing))
                    for layer, gate in self._routed_gates
                ]
        if self.token_loss or self.experts or dumps_enabled:
            self._loss_observer = _GuardedObserver(self._observe_loss, self, "loss_observer")

    def _install_pre_reduce_observers(self, pre_reduce_targets: _PreReduceTargets) -> None:
        self.pre_reduce_observers = []
        for super_tensor, metric_name, layer, comm_stream in pre_reduce_targets:
            observer = PreReduceGradObserver(self.recorder, metric_name, layer, comm_stream)
            self._pre_reduce_guards.append(
                (super_tensor, _GuardedObserver(observer, self, f"pre_reduce_observer(layer={layer})")))
            self.pre_reduce_observers.append(observer)
            self._pre_reduce_super_tensors.append(super_tensor)

    @property
    def dump_capture(self) -> DumpCapture | None:
        return None if self.rank_capture is None else self.rank_capture.dump_capture

    def _captures(self, step: int) -> bool:
        return step % self.capture_interval == 0

    def _set_step_observers(self, active: bool) -> None:
        """Attaches the observers the model and the monitors call on their own -- routing, loss, pre-reduce gradients
        and the rank-local monitors' -- for a captured step, or detaches them for a step that is not captured, so that
        the model and the monitors skip that step's capture work. Monitors listed before this callback start a step
        before it does, so the observers of a step are set when the step before it ends."""
        if active == self._step_observers_active:
            return
        self._step_observers_active = active
        for gate, observer in self._routing_observers:
            gate.routing_observer = observer if active else None
        self.causal_lm.loss_observer = self._loss_observer if active else None
        for super_tensor, observer in self._pre_reduce_guards:
            super_tensor.pre_reduce_observer = observer if active else None
        if self.rank_local_monitors:
            self.monitor_capture.attach(active)

    def disable(self, error: Exception, where: str) -> None:
        """Turns every subsequent hook/observer call into a no-op after metric capture's first
        recoverable failure. Exhausting capture capacity instead stops training before losing
        a step. Already-written windows stay valid."""
        if self.disabled:
            return
        self.disabled = True
        log.exception("metric_sampler: disabling metric capture after a failure in %s", where)
        # The node process then stops waiting for this rank's steps instead of timing out on each.
        if self.rank_capture is not None:
            self.rank_capture.ring.request_stop()
        for _, gate in self._routed_gates:
            gate.routing_observer = None
        self.monitor_capture.detach_expert_observers()
        for _, dispatcher in self.dispatchers:
            dispatcher.assignment_observer = None
        self._set_attention_observers(False)
        if self.causal_lm is not None:
            self.causal_lm.loss_observer = None
        for super_tensor in self._pre_reduce_super_tensors:
            super_tensor.pre_reduce_observer = None
        global_actlog_state.activation_observer = None
        self._disable_dumps(error, where, log_it=False)

    def _disable_dumps(self, error: Exception, where: str, log_it: bool = True) -> None:
        if self.dump_capture is None:
            return
        if log_it:
            log.exception("metric_sampler: disabling activation dumps after a failure in %s", where)
        self.rank_capture.stop_dumps()

    def _guard(self, where: str, function: Callable[..., None], *args: object) -> None:
        try:
            function(*args)
        except (ActivationDumpConfigError, CaptureCapacityError):
            raise
        except Exception as error:
            if self.failures_stop_training:
                raise
            self.disable(error, where)

    def _guard_dump(self, where: str, function: Callable[..., None], *args: object) -> None:
        try:
            function(*args)
        except Exception as error:
            self._disable_dumps(error, where)

    def batch_start(self, state: State, logger: Logger):
        if self.disabled:
            return
        self._guard("batch_start", self._batch_start, state, logger)

    def _batch_start(self, state: State, logger: Logger):
        if self.rank_capture is not None and not self.rank_capture.node_running():
            raise CaptureCapacityError("metric sampler node writer exited; capture cannot continue")
        step = state.timestamp.batch.value + 1
        now = time.perf_counter()
        self._capturing = self._captures(step)
        if not self._capturing:
            # The next captured step's loop time is that of the step before it, as on every step.
            self.step_started = now
            return
        # The attempt's first step is always detailed so every Triton kernel variant compiles
        # here instead of stalling training on whichever later step first goes detailed.
        detailed = self._first_batch_start or self.recorder.verbose(step) or step % self.distribution_interval == 0
        self._first_batch_start = False
        self.recorder.begin_step(detailed)
        self.token_loss_stats.begin_step(detailed)
        self.activation_stats.begin_step(detailed)
        if self.attention_stats is not None:
            self.attention_stats.begin_step(detailed)
        self.microbatch = 0
        self.monitor_capture.begin_step()
        if self.step_started:
            self._record_step_scalar("time/loop_s", now - self.step_started)
        self.step_started = now
        self._step_start_event = self._recorded_timing_event()
        self._guard_dump("batch_start", self._dump_batch_start, step)

    def _dump_batch_start(self, step: int) -> None:
        if self.dump_capture is None:
            return
        self.dump_capture.begin_step(step)
        self._record_step_scalar("sampler/dumps_skipped", float(self.dump_capture.dumps_skipped))
        if self.dump_capture.budget_exhausted and not self._dump_budget_logged:
            self._dump_budget_logged = True
            log.warning("metric_sampler: activation dump max_gb budget reached, dumps stopped for this attempt")

    def before_forward(self, state: State, logger: Logger):
        if self.disabled or not self._capturing:
            return
        self._guard("before_forward", self._before_forward, state, logger)

    def _before_forward(self, state: State, logger: Logger):
        stream_ids = state.batch.get("stream_id")
        if stream_ids is not None and stream_ids.shape != (state.batch["input_ids"].shape[0],):
            raise ValueError("metric_sampler requires one stream_id per microbatch sample")
        self._capture_forward_active = True
        self._monitor_observations = None
        self.activation_stats.begin_microbatch(self.microbatch)
        global_actlog_state.activation_observer = self._activation_observer
        self.token_loss_stats.begin_microbatch(
            self.microbatch, stream_ids, state.batch.get("sample_ids")
        )
        for dispatcher, observer in self._dispatcher_observers:
            dispatcher.assignment_observer = observer
        self._set_attention_observers(True)
        self._forward_start_event = self._recorded_timing_event()
        self._guard_dump("before_forward", self._dump_before_forward, state)

    def _begin_model_pass(self, module, args) -> None:
        # GigaFSDP's first call runs the model twice: a no-grad pass that records module order,
        # then the training pass. Only the pass with grad enabled is observed.
        self._grad_forward = torch.is_grad_enabled()
        if self.disabled or not self._capture_forward_active:
            return
        global_actlog_state.activation_observer = self._activation_observer if self._grad_forward else None

    def forward_is_captured(self) -> bool:
        """Whether the forward in progress is a training forward with grad that the sampler captures, and not the
        recomputation of a checkpointed forward during backward."""
        return (self._capture_forward_active and self._grad_forward
                and torch._C._current_graph_task_id() == _OUTSIDE_BACKWARD)

    def _dump_before_forward(self, state: State) -> None:
        if self.dump_capture is None:
            return
        self.dump_capture.begin_microbatch(self.microbatch, state.batch.get("stream_id"))
        if self.microbatch == 0:
            self.dump_capture.observe_input_ids(state.batch["input_ids"])

    def after_forward(self, state: State, logger: Logger):
        if self.disabled or not self._capturing:
            return
        self._guard("after_forward", self._after_forward, state, logger)

    def _after_forward(self, state: State, logger: Logger):
        self._capture_forward_active = False
        if self.rank_local_monitors:
            self.monitor_capture.after_forward()
        if self.quantile_stats is not None:
            self.quantile_stats.flush()
            self.activation_stats.flush()
        # Checkpoint recomputation must not replace the original forward's measurements.
        global_actlog_state.activation_observer = None
        for dispatcher, _ in self._dispatcher_observers:
            dispatcher.assignment_observer = None
        self._set_attention_observers(False)
        self.recorder.record_gpu_duration(
            "time/forward_gpu_s", NO_LAYER, self.microbatch, self._forward_start_event, self._recorded_timing_event()
        )

    def _observe_dispatcher(
        self, layer: int, before: torch.Tensor, valid_before: torch.Tensor,
        after: torch.Tensor, valid_after: torch.Tensor,
    ) -> None:
        counts = (before, valid_before, after, valid_after)
        if all(count.is_cuda and count.dtype in (torch.int64, torch.int32) and count.is_contiguous()
               for count in counts):  # pragma: no cover
            extension = activation_ext.prebuilt()
            if extension:
                slots = self._dispatcher_slots(layer)
                for name, count, slot in zip(DISPATCHER_METRICS, counts, slots):
                    if count.numel() != slot.width:
                        raise ValueError(f"{name} expected {slot.width} values, got {count.numel()}")
                self.recorder.mark_recorded([slot.slot_index for slot in slots])
                extension.copy_counts_into(list(counts), self.recorder.buffer, [slot.offset for slot in slots])
                return
        for name, count in zip(DISPATCHER_METRICS, counts):
            self.recorder.record(name, layer, self.microbatch, count)

    def _dispatcher_slots(self, layer: int) -> tuple[ResolvedSlot, ...]:  # pragma: no cover
        key = (layer, self.microbatch)
        slots = self._dispatcher_count_slots.get(key)
        if slots is None:
            slots = self._dispatcher_count_slots[key] = tuple(
                self.recorder.resolve(name, layer, self.microbatch) for name in DISPATCHER_METRICS)
        return slots

    @staticmethod
    def _attention_gate(decoder_layer: torch.nn.Module) -> torch.nn.Module:
        """An MTP head's self_attn is wrapped in MTPAttnAdapter, which forwards to the AttnGate as base_attn; a main
        trunk self_attn already is the AttnGate (as expert_balancing_monitor.get_moe_gates unwraps it)."""
        return getattr(decoder_layer.self_attn, "base_attn", decoder_layer.self_attn)

    def _find_attention_layers(self, mtp_decoder_layers: _DecoderLayers) -> None:
        for layer, decoder_layer in self._decoder_layers(mtp_decoder_layers):
            # A MoE layer's attention is wrapped in AttnGate.
            attention_gate = self._attention_gate(decoder_layer)
            attention = getattr(attention_gate, "_attn", attention_gate)
            if isinstance(attention, LlamaLatentAttention):
                self._softmax_attention.append((layer, attention))
            elif isinstance(attention, Qwen3NextGatedDeltaNet):
                self._linear_attention.append((layer, attention))
        if not self._softmax_attention and not self._linear_attention:
            raise ValueError("attention_stats requires a LlamaLatentAttention or Qwen3NextGatedDeltaNet layer")

    def _build_attention_observers(self) -> None:
        native = self.attention_stats.native_calls(self)
        if native is not None:  # pragma: no cover
            # The extension's observers take a call without a Python call; an MLA call's rows still go to Python.
            self._attention_observers = [
                (attention, native.softmax(
                    layer, attention.qk_nope_head_dim, functools.partial(self._observe_softmax_rows, layer)))
                for layer, attention in self._softmax_attention
            ] + [(attention, native.linear(layer)) for layer, attention in self._linear_attention]
            self._gate_observers = [
                (attention, native.gate(layer)) for layer, attention in self._softmax_attention
                if attention.gated_attention
            ]
            return
        self._attention_observers = [
            (attention, self.layer_observer(
                "attention_observer", layer, self._observe_softmax_attention, attention.qk_nope_head_dim))
            for layer, attention in self._softmax_attention
        ] + [
            (attention, self.layer_observer("attention_observer", layer, self._observe_linear_attention))
            for layer, attention in self._linear_attention
        ]
        self._gate_observers = [
            (attention, self.layer_observer("gate_observer", layer, self._observe_softmax_gate))
            for layer, attention in self._softmax_attention
            if attention.gated_attention
        ]

    def _set_attention_observers(self, active: bool) -> None:
        for attention, observer in self._attention_observers:
            attention.attention_observer = observer if active else None
        for attention, observer in self._gate_observers:
            attention.gate_observer = observer if active else None

    def _observe_softmax_attention(
        self, layer: int, nope_dim: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        lse: torch.Tensor, softmax_scale: float, cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> None:
        if self.forward_is_captured():
            self.attention_stats.observe_softmax(layer, self.microbatch, query, key, value, lse, nope_dim)
            self._observe_softmax_rows(layer, query, key, value, lse, softmax_scale, cu_seqlens, max_seqlen)

    def _observe_softmax_rows(
        self, layer: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, lse: torch.Tensor,
        softmax_scale: float, cu_seqlens: torch.Tensor | None = None, max_seqlen: int | None = None,
    ) -> None:
        if cu_seqlens is None:
            lse = lse.transpose(0, 1)
        else:
            query, key = query.unsqueeze(0), key.unsqueeze(0)
        self.attention_stats.observe_softmax_rows(
            layer, self.microbatch,
            functools.partial(attention_row_stats, query, key, lse, softmax_scale, cu_seqlens, max_seqlen),
        )

    def _observe_softmax_gate(self, layer: int, gate: torch.Tensor) -> None:
        if self.forward_is_captured():
            self.attention_stats.observe_softmax_gate(layer, self.microbatch, gate)

    def _observe_linear_attention(
        self, layer: int, beta: torch.Tensor, log_decay: torch.Tensor, value: torch.Tensor, output: torch.Tensor,
        a_log: torch.Tensor, dt_bias: torch.Tensor, final_state: torch.Tensor | None,
    ) -> None:
        if self.forward_is_captured():
            self.attention_stats.observe_linear(layer, self.microbatch, beta, log_decay, value, output, a_log, dt_bias)
            if final_state is not None:
                self.attention_stats.observe_linear_state(layer, self.microbatch, final_state)

    def _observe_monitor_module(self, name: str, module, inputs, output) -> None:
        observations = self.monitor_capture.observe_module(name, module, inputs, output)
        if observations is not None:  # pragma: no cover
            self._monitor_observations = observations

    def monitor_hook(self, name: str) -> Callable:
        """The forward hook of the monitored module `name`: the extension's, which takes a call without Python once
        the forward has its MonitorObservations, or the Python one."""
        extension = activation_ext.prebuilt()
        if extension:  # pragma: no cover
            return extension.MonitorModuleHook(self, name, torch._C._current_graph_task_id)
        return functools.partial(self._observe_monitor_module, name)

    def _monitor_slots(self, name: str, inputs, outputs) -> list[int]:  # pragma: no cover
        return self.monitor_capture.module_slots(name, inputs, outputs)

    def _observe_routing(self, layer: int, topk_indices: torch.Tensor, scores: torch.Tensor) -> None:
        self.token_loss_stats.observe_routing(layer, topk_indices, scores)
        if self.quantile_stats is None:
            return
        scores = scores.detach()
        if (scores.is_cuda and scores.dtype == torch.float32 and scores.ndim == _ROUTER_SCORES_NDIM and scores.numel()
                and scores.stride(-1) == 1):  # pragma: no cover
            extension = activation_ext.prebuilt()
            if extension:
                # One extension call reads the scores in place and writes the quantiles into their slots.
                quantiles, counts = self._entropy_slots(layer)
                self.quantile_stats.claim("router/entropy_quantiles", layer, self.microbatch)
                self.recorder.mark_recorded([quantiles.slot_index, counts.slot_index])
                if not extension.entropy_quantiles_into(scores, self.recorder.buffer, quantiles.offset, counts.offset):
                    self._copy_entropy_quantiles(layer, scores)
                return
        self._record_entropy_quantiles(layer, scores)

    def _copy_entropy_quantiles(self, layer: int, scores: torch.Tensor) -> None:  # pragma: no cover
        """The exact entropy quantiles of scores too many for the extension's selection, into the claimed slots."""
        quantiles, counts = self._entropy_slots(layer)
        selected, population = self._entropy_quantiles(scores)
        self.recorder.view(quantiles).copy_(selected.view(torch.int32))
        self.recorder.view(counts).copy_(population)

    def _record_entropy_quantiles(self, layer: int, scores: torch.Tensor) -> None:
        quantiles, counts = self._entropy_quantiles(scores)
        self.quantile_stats.record_float64(
            "router/entropy_quantiles", "router/entropy_quantile_counts", layer, self.microbatch, quantiles, counts,
        )

    def _native_routing(self, experts: int) -> "_NativeRouting | None":  # pragma: no cover
        """The extension's routing observers, which take a call as _observe_routing does, without a Python call.
        None without the extension, over a host recorder, or where the guard of the Python observers would disable
        the sampler on a failure instead of raising it."""
        extension = activation_ext.prebuilt()
        if not extension or not self.recorder.buffer.is_cuda or not self.failures_stop_training:
            return None
        return extension.RoutingCalls(
            self, self.token_loss_stats, self.quantile_stats, self.recorder, self._native_entropy_slots,
            self._copy_entropy_quantiles, self._record_entropy_quantiles, self._observe_routing, triton_launch,
            token_loss_kernel, experts, self.recorder.recorded_flags(),
        )

    def _native_entropy_slots(self, layer: int, microbatch: int) -> tuple[int, int, int, int]:  # pragma: no cover
        """The (quantiles offset, quantiles slot index, counts offset, counts slot index) of a router's entropy
        quantiles in a call RoutingCalls takes."""
        quantiles = self.recorder.resolve("router/entropy_quantiles", layer, microbatch)
        counts = self.recorder.resolve("router/entropy_quantile_counts", layer, microbatch)
        return quantiles.offset, quantiles.slot_index, counts.offset, counts.slot_index

    def _entropy_slots(self, layer: int) -> tuple[ResolvedSlot, ResolvedSlot]:  # pragma: no cover
        key = (layer, self.microbatch)
        slots = self._entropy_quantile_slots.get(key)
        if slots is None:
            slots = self._entropy_quantile_slots[key] = (
                self.recorder.resolve("router/entropy_quantiles", layer, self.microbatch),
                self.recorder.resolve("router/entropy_quantile_counts", layer, self.microbatch),
            )
        return slots

    @staticmethod
    def _entropy_quantiles(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact quantiles of the routed tokens' normalized float64 score entropy."""
        return exact_quantiles(row_entropy(scores))

    def _dump_observe(self, point: str, layer: int, tensor: torch.Tensor) -> None:
        self._guard_dump("activation_observer(dump)", self._dump_observe_point, point, layer, tensor)

    def _dump_observe_point(self, point: str, layer: int, tensor: torch.Tensor) -> None:
        if self.dump_capture is not None:
            self.dump_capture.observe(point, layer, tensor)

    def _observe_loss(self, loss: torch.Tensor, labels_valid: torch.Tensor) -> None:
        self.token_loss_stats.observe_loss(loss, labels_valid)
        if self.quantile_stats is not None and self.token_loss:
            self.quantile_stats.observe(
                "loss/token_quantiles", "loss/token_quantile_counts", NO_LAYER, self.microbatch,
                loss[0], labels_valid[0],
            )
            for segment, layer in enumerate(self._mtp_layers, start=1):
                self.quantile_stats.observe(
                    "loss/mtp_token_quantiles", "loss/mtp_token_quantile_counts", layer, self.microbatch,
                    loss[segment], labels_valid[segment],
                )
        self._guard_dump("loss_observer", self._dump_observe_loss, loss, labels_valid)

    def _dump_observe_loss(self, loss: torch.Tensor, labels_valid: torch.Tensor) -> None:
        if self.dump_capture is not None:
            self.dump_capture.observe_loss(loss, labels_valid)

    def after_loss(self, state: State, logger: Logger):
        if self.disabled or not self._capturing:
            return
        self._guard("after_loss", self._after_loss, state, logger)

    def _after_loss(self, state: State, logger: Logger):
        for key, value in state.loss.items():
            name = LOSS_METRICS.get(key)
            if name is None:
                if key not in self.unknown_loss_keys:
                    self.unknown_loss_keys.add(key)
                    log.warning("metric_sampler: loss component %s has no registered metric and is not recorded", key)
                continue
            self.recorder.record(name, NO_LAYER, self.microbatch, value)

    def before_backward(self, state: State, logger: Logger):
        if self.disabled or not self._capturing:
            return
        self._guard("before_backward", self._before_backward, state, logger)

    def _before_backward(self, state: State, logger: Logger):
        self._backward_start_event = self._recorded_timing_event()

    def after_backward(self, state: State, logger: Logger):
        if self.disabled or not self._capturing:
            return
        self._guard("after_backward", self._after_backward, state, logger)

    def _after_backward(self, state: State, logger: Logger):
        if self.rank_local_monitors:
            self.monitor_capture.after_backward()
        self.recorder.record_gpu_duration(
            "time/backward_gpu_s", NO_LAYER, self.microbatch, self._backward_start_event, self._recorded_timing_event()
        )
        if self.microbatch == self.microbatches - 1:
            # UpdateBias consumes and zeroes local_tokens_per_expert on AFTER_TRAIN_BATCH, and
            # algorithms run before callbacks, so the step totals are only intact here.
            for layer, gate in self.counting_gates:
                self.recorder.record(
                    "router/local_tokens_before_bias_update", layer, NO_MICROBATCH, gate.local_tokens_per_expert
                )
                self.recorder.record("router/expert_bias_before_update", layer, NO_MICROBATCH, gate.expert_bias)
        self.microbatch += 1

    def after_train_batch(self, state: State, logger: Logger):
        if self.disabled or not self._capturing:
            return
        self._guard("after_train_batch", self._after_train_batch, state, logger)

    def _after_train_batch(self, state: State, logger: Logger):
        self._optimizer_start_event = self._recorded_timing_event()

    def batch_end(self, state: State, logger: Logger):
        if self.disabled:
            return
        if self._capturing:
            self._guard("batch_end", self._batch_end, state, logger)
        if not self.disabled:
            self._set_step_observers(self._captures(state.timestamp.batch.value + 1))

    def _batch_end(self, state: State, logger: Logger):
        # The pre-reduce observers ran on GigaFSDP's comm stream; wait for them explicitly
        # instead of relying on GigaFSDP's own internal synchronization.
        for observer in self.pre_reduce_observers:
            if observer.event is not None:
                torch.cuda.current_stream().wait_event(observer.event)
                observer.event = None

        if self.gradient_clipping is not None and isinstance(self.gradient_clipping.last_total_norm, torch.Tensor):
            self.recorder.record(
                "grad/clip_total_norm", NO_LAYER, NO_MICROBATCH, self.gradient_clipping.last_total_norm
            )

        self.recorder.record_gpu_duration(
            "time/optimizer_gpu_s", NO_LAYER, NO_MICROBATCH, self._optimizer_start_event, self._recorded_timing_event()
        )
        self.recorder.record_gpu_duration(
            "time/step_gpu_s", NO_LAYER, NO_MICROBATCH, self._step_start_event, self._recorded_timing_event()
        )

        memory_stats = torch.cuda.memory_stats()
        self._record_step_scalar("memory/allocated_gb", memory_stats["allocated_bytes.all.current"] / _BYTES_PER_GB)
        self._record_step_scalar("memory/peak_allocated_gb", memory_stats["allocated_bytes.all.peak"] / _BYTES_PER_GB)
        self._record_step_scalar("memory/reserved_gb", memory_stats["reserved_bytes.all.current"] / _BYTES_PER_GB)
        self._record_step_scalar("memory/alloc_retries", memory_stats["num_alloc_retries"])

        scalars = {
            "time/step_s": time.perf_counter() - self.step_started,
            "lr": state.optimizers[0].param_groups[0]["lr"],
            "sampler/dropped_steps": self.recorder.dropped_steps,
            "sampler/microbatches": self.microbatch,
            "sampler/step_end_s": self.last_step_end_s,
            "sampler/ring_used_slots": self.recorder.ring_used_slots,
            "sampler/ring_capacity_slots": self.recorder.ring_capacity_slots,
        }
        for name, value in scalars.items():
            self._record_step_scalar(name, float(value))
        if self.rank_local_monitors:
            self.monitor_capture.end_step()
        started = time.perf_counter()
        self.recorder.step_end(state.timestamp.batch.value)
        self.last_step_end_s = time.perf_counter() - started
        self._guard_dump("batch_end", self._dump_step_end)

    def _record_step_scalar(self, metric_name: str, value: float) -> None:
        self.recorder.record_host(metric_name, NO_LAYER, NO_MICROBATCH, value)

    @staticmethod
    def _recorded_timing_event() -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _dump_step_end(self) -> None:
        if self.dump_capture is not None:
            self.dump_capture.step_end()

    def fit_end(self, state: State, logger: Logger):
        self.monitor_capture.finish()
        self._shutdown()

    def close(self, state: State, logger: Logger):
        self._shutdown()

    def _shutdown(self):
        self._set_attention_observers(False)
        self.attention_stats = None
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.monitor_capture.shutdown()
        self._monitor_observations = None
        self._entropy_quantile_slots.clear()
        self._dispatcher_count_slots.clear()

        for _, dispatcher in self.dispatchers:
            dispatcher.assignment_observer = None
        if self.activation_stats is not None:
            global_actlog_state.activation_observer = None
            self.activation_stats = None
        rank_capture, self.rank_capture = self.rank_capture, None
        self.recorder = None
        if rank_capture is not None:
            rank_capture.close()

    def _gigafsdp_pre_reduce_targets(self, state: State, num_layers: int) -> _PreReduceTargets:
        # Use the wrapped MoE block's type, matching special_process_group_fn, even when
        # expert and dense parameters share the same process group at EP=1.
        if not state.gigafsdp_enabled:
            return []
        # giga_fsdp is installed only where GigaFSDP trains.
        from giga_fsdp import GigaFSDP, SuperTensorModule

        targets = []
        for module in state.model.modules():
            if not isinstance(module, GigaFSDP):
                continue
            if module._zero_stage != _PRE_REDUCE_ZERO_STAGE:
                raise ValueError(
                    "metric_sampler pre-reduce gradient statistics currently require GigaFSDP zero_stage=1"
                )
            # A distinct process group is not a reliable dense/expert discriminator: with EP=1,
            # special_process_group_fn deliberately assigns MoE blocks the default group.
            super_tensor_roots = {
                id(st_module.super_tensor): st_module.root_module
                for st_module in module._module.modules()
                if isinstance(st_module, SuperTensorModule)
            }
            for super_tensor in module._super_tensors:
                if not super_tensor.is_trainable:
                    continue
                layer = self._layer_from_super_tensor_name(super_tensor.name, num_layers)
                root_module = super_tensor_roots.get(id(super_tensor))
                if root_module is None and super_tensor is not getattr(module, "_layernorm_supertensor", None):
                    raise ValueError(
                        f"metric_sampler could not determine the owning module for trainable SuperTensor "
                        f"{super_tensor.name!r}"
                    )
                is_expert = isinstance(root_module, (AbstractGMMMoeBlock, ScMoEBlock))
                metric_name = "grad/pre_reduce_sq_expert" if is_expert else "grad/pre_reduce_sq_dense"
                targets.append((super_tensor, metric_name, layer, module._comm_stream))
        return targets

    def _layout_requests(
        self,
        experts: int,
        routed_layers: dict[int, int],
        mtp_layers: list[int],
        streams: int,
        pre_reduce_keys: set[tuple[str, int]],
        activation_layers: dict[str, list[int]],
        dumps_enabled: bool,
    ) -> list[tuple[MetricSpec, int, int, int]]:
        return LayoutPlan(
            microbatches=self.microbatches,
            routed_experts=experts,
            streams=streams,
            loss_metrics=tuple(LOSS_METRICS.values()),
            counting_gate_layers=tuple(layer for layer, _ in self.counting_gates),
            dispatcher_layers=tuple(layer for layer, _ in self.dispatchers),
            routed_layers=tuple(routed_layers),
            mtp_layers=tuple(mtp_layers),
            pre_reduce_keys=frozenset(pre_reduce_keys),
            activation_layers={point: tuple(layers) for point, layers in activation_layers.items()},
            softmax_attention=tuple(
                SoftmaxAttentionLayer(layer, attention.num_heads_per_rank, attention.gated_attention)
                for layer, attention in self._softmax_attention
            ),
            linear_attention=tuple(
                LinearAttentionLayer(layer, attention.num_v_heads) for layer, attention in self._linear_attention
            ),
            token_loss=self.token_loss,
            experts=self.experts,
            activations=self.activations,
            exact_quantiles=self.exact_quantiles,
            dumps=dumps_enabled,
            extra=tuple(self.monitor_capture.slot_requests(experts, routed_layers)) if self.rank_local_monitors else (),
        ).requests()

    @staticmethod
    def _experts_module(decoder_layer: torch.nn.Module) -> torch.nn.Module | None:
        block = getattr(decoder_layer, "block_sparse_moe", None)
        experts = getattr(block, "experts", None)
        if experts is None:
            experts = getattr(getattr(block, "gemm", None), "experts", None)
        return experts

    def _activation_point_layers(self, mtp_decoder_layers: _DecoderLayers) -> dict[str, list[int]]:
        # Decided from the real module tree, not config flags, so a point never gets a Layout
        # slot for a layer that can't call observe for it (KeyError on the first activation step).
        # shared_expert_out needs a shared_experts submodule; swiglu_in/down_in fire only for GroupedLlamaMLP experts
        # (used by DeepseekGMMMoeBlock, ScMoEBlock/ScMoEBlockGeMM, but not SonicGMMMoeBlock's GroupedSonicMLP).
        layers: dict[str, list[int]] = {name: [] for name in LAYER_ACTIVATION_POINTS}
        for layer, decoder_layer in self._decoder_layers(mtp_decoder_layers):
            layers["residual"].append(layer)
            layers["attn_out"].append(layer)
            layers["attn_residual_contribution"].append(layer)
            layers["mlp_residual_contribution"].append(layer)
            layers["mlp_out"].append(layer)
            if getattr(self._attention_gate(decoder_layer), "shared_experts", None) is not None:
                layers["shared_expert_out"].append(layer)
            if isinstance(self._experts_module(decoder_layer), GroupedLlamaMLP):
                layers["swiglu_in"].append(layer)
                layers["down_in"].append(layer)
        return layers

    @staticmethod
    def _layer_index(tag: str, num_layers: int) -> int:
        kind, index = tag.rsplit("_", 1)
        return int(index) if kind == "layer" else num_layers + int(index)

    @staticmethod
    def _layer_from_super_tensor_name(name: str, num_layers: int) -> int:
        match = _MAIN_LAYER_NAME_PATTERN.match(name)
        if match:
            return int(match.group(1))
        match = _MTP_LAYER_NAME_PATTERN.match(name)
        if match:
            return num_layers + int(match.group(1))
        return NO_LAYER
