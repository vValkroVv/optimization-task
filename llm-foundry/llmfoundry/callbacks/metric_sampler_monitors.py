from __future__ import annotations

import functools
import json
import os
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Protocol

import torch
from composer.callbacks.activation_monitor import ActivationMonitor
from composer.callbacks.lr_monitor import LRMonitor
from composer.callbacks.memory_monitor import MemoryMonitor
from composer.callbacks.speed_monitor import SpeedMonitor
from composer.callbacks.weight_monitor import WeightMonitor, cleanup_metric_name, is_internal_parameter_name
from composer.callbacks.optimizer_monitor import OptimizerMonitor
from composer.callbacks.training_metrics_monitor import TrainingMetricsMonitor
from composer.core import Callback, State
from composer.utils import dist

from llmfoundry.callbacks.metric_sampler_weights import MATERIALIZED_WEIGHT_ROW_WIDTH, MaterializedWeights
from llmfoundry.callbacks.optimizer_health_monitor import OptimizerHealthMonitor, PARAM_GROUPS
from llmfoundry.callbacks.stage_time_profiling import StageTimeProfiler
from llmfoundry.callbacks.update_weight_ratio_monitor import N_ACCUM_VALUES, UpdateWeightRatioMonitor
from llmfoundry.callbacks.topk_accuracy_monitor import TopkAccuracyMonitor
from llmfoundry.callbacks.router_margin_monitor import RouterMarginMonitor
from llmfoundry.models.layers.moe.experts import GroupedLlamaMLP
from llmfoundry.models.layers.moe.routers import MoEGate
from llmfoundry.models.ops.float8.fp8_tensor import Float8BlockwiseQTensor
from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.kernels.graphed import GraphedComputation
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder, ResolvedRows, ResolvedSlot
from metric_sampler.capture.stats.expert_balance import ExpertBalanceStats
from metric_sampler.capture.stats.monitor_activations import MonitorActivationStats, MonitorObservations
from metric_sampler.capture.stats.router_margin import (
    ROUTER_MARGIN_WIDTH,
    router_margin_graph_stats,
    router_margin_stats,
)
from metric_sampler.contract.metrics import LAYER_IDENTITY_LIMIT, NO_LAYER, NO_MICROBATCH, WORDS_PER_FLOAT64

if TYPE_CHECKING:
    from llmfoundry.callbacks.metric_sampler import MetricSampler

__all__ = ["RankLocalMonitorCapture"]

# The rank of a catalog entry that holds on every rank.
_ALL_RANKS = -1
# A rank-local counter is recorded as its low and its high UInt32 word; an absent value as zero words.
_UINT32_BITS = 32
_UINT32_MASK = 0xFFFFFFFF
_UINT64_LIMIT = 2**64
_ABSENT_WORDS = (0,) * WORDS_PER_FLOAT64

# What the monitors record, in the order of their values.
_TOPK_K_VALUES = (1, 5, 20)
_OPTIMIZER_METRIC_PREFIXES = (
    "l2_norm/grad", "l2_norm/moment", "l2_norm/moment_sq", "l2_norm/param", "l2_norm/update",
    "l2_norm/moment_bc", "l2_norm/moment_sq_bc", "amax/moment", "amax/moment_sq",
)
_OPTIMIZER_HEALTH_FIELD_ORDER = (
    "numel", "frac_eps_count", "frac_10eps_count", "update_abs_sum", "grad_contrib_sq_sum", "wd_contrib_sq_sum",
)
_ROUTER_BALANCE_ORDER = (
    "normalized_topk_sum", "raw_topk_sum", "biased_topk_sum", "ordinary_biased_score_entropy_sum",
    "router_z_loss_sum", "seq_aux_loss_sum", "alpha_sum",
)
_ROUTER_BALANCE_COUNTS_ORDER = ("all_tokens", "valid_tokens", "forward_calls")
_TRAINING_COUNTER_ORDER = ("skip_updates", "total_updates", "zeroed_module_grads", "processed_module_grads")
_TRAINING_LOCAL_ORDER = ("rank_samples", "rank_tokens", "rank_batch_seconds", "device_flops")

# Slot widths of the monitors. A value recorded as its Int32 words is followed by a presence flag.
_VALUE_WITH_PRESENCE_WIDTH = WORDS_PER_FLOAT64 + 1
_OPTIMIZER_HEALTH_WIDTH = len(PARAM_GROUPS) * len(_OPTIMIZER_HEALTH_FIELD_ORDER) * WORDS_PER_FLOAT64
# A float32 value and its presence flag per optimizer metric.
_OPTIMIZER_LOCAL_WIDTH = 2 * len(_OPTIMIZER_METRIC_PREFIXES)
# The absolute max and the L2 norm.
_WEIGHT_STATISTICS_WIDTH = 2
# The absolute max and the row-weighted mean L2 norm, or the observation count and the row count.
_ACTIVATION_LOCAL_WIDTH = 2
# The valid labels and the hits at each k.
_TOPK_COUNTS_WIDTH = 1 + len(_TOPK_K_VALUES)
_UPDATE_WEIGHT_WIDTH = N_ACCUM_VALUES * WORDS_PER_FLOAT64
_ROUTER_BALANCE_WIDTH = len(_ROUTER_BALANCE_ORDER) * WORDS_PER_FLOAT64


class _NativeRouterMargins(Protocol):
    """The extension's RouterMarginCalls: the routed layers' router margin hooks, made from C++."""

    def router(
        self, layer: int, scoring_function: str, top_k: int, near_count: int, experts: int,
        record: Callable[[torch.Tensor], None],
    ) -> Callable[..., None]: ...


class RankLocalMonitorCapture:
    """Captures what the trainer's monitors, the routers and the experts see on this rank before any reduction.

    `prepare` finds the monitors the sampler needs among the callbacks listed before it and agrees the catalogs with
    the other ranks; `install` makes the observers and the hooks; `attach` hands them to the monitors, the routers and
    the experts for a captured step and takes them back for any other; the step's flushes record what the observers
    gathered. The sampler gives the recorder, the microbatch and the guard of its observers."""

    def __init__(self, sampler: MetricSampler) -> None:
        self._sampler = sampler
        self._routed_gates: list[tuple[int, MoEGate]] = []
        self._expert_modules: list[tuple[int, GroupedLlamaMLP]] = []
        self._handles: list = []
        self._parameter_names: list[str] = []
        self._update_parameter_names: list[str] = []
        self._optimizer_parameter_names: list[str] = []
        self._parameter_indices: dict[str, int] = {}
        self._resolved_parameter_rows: dict[tuple[str, tuple[str, ...]], ResolvedRows] = {}
        # The presence pattern of the last call, its presence payload and the positions of its emitted metrics.
        self._optimizer_presence: tuple[tuple[bool, ...], torch.Tensor, torch.Tensor] | None = None
        self._optimizer_keys: tuple[tuple[str, ...], ...] = ()
        self._lr_monitor = None
        self._memory_monitor = None
        self._speed_monitor = None
        self._stage_time_monitor = None
        self._weight_monitor = None
        self._optimizer_monitor = None
        self._training_monitor = None
        self._optimizer_health_monitor = None
        self._update_weight_monitor = None
        self._topk_monitor = None
        # The trainer state, which holds the observer of the training counters.
        self._trainer_state: State | None = None
        self._lr_group_names: list[str] = []
        self._lr_group_indices: dict[str, int] = {}
        self._materialized_parameter_names: list[str] = []
        # Each module inside a SuperTensorModule, mapped to it.
        self._materialized_units: dict[torch.nn.Module, torch.nn.Module] = {}
        self.materialized_weights: MaterializedWeights | None = None
        # The weight monitor's observer: the extension's, or _observe_materialized_weights.
        self._weight_observer: Callable[[torch.nn.Module], None] = self._observe_materialized_weights
        self._training_local: tuple[int, int, float] | None = None
        self._local_flops: float | None = None
        self._activation_modules: list[tuple[torch.nn.Module, str]] = []
        self._log_output = False
        self._log_gradients = False
        self._log_grad_input = False
        self._activation_catalog: list[str] = []
        self._catalog: list[dict[str, int | str]] = []
        self.monitor_activations: MonitorActivationStats | None = None
        self.expert_balance_stats: ExpertBalanceStats | None = None
        self._balance_observers: list[tuple[MoEGate, object]] = []
        self._expert_activation_observers: list[tuple[GroupedLlamaMLP, object]] = []
        # Per (layer, microbatch): the slot of the router margins.
        self._router_margin_slots: dict[tuple[int, int], ResolvedSlot] = {}
        self._router_margin_layers: set[int] = set()
        self._router_margin_graphs: dict[tuple, GraphedComputation] = {}

    @property
    def _recorder(self) -> RankRecorder:
        return self._sampler.recorder

    @property
    def _microbatch(self) -> int:
        return self._sampler.microbatch

    def prepare(
        self, state: State, routed_gates: list[tuple[int, MoEGate]], expert_modules: list[tuple[int, GroupedLlamaMLP]]
    ) -> None:
        """Finds the monitors listed before the sampler and agrees the catalogs of parameters, groups and activations
        with the other ranks. `routed_gates` and `expert_modules` are the layers the sampler found in the model."""
        self._routed_gates = routed_gates
        self._expert_modules = expert_modules
        if sys.byteorder != "little":
            raise ValueError("rank-local Float64 payloads require little-endian words")
        self._prepare_optimizer_monitors(state)
        self._prepare_monitor_additions(state)
        self._parameter_names = sorted(set(self._optimizer_parameter_names) | set(self._materialized_parameter_names))
        self._parameter_indices = {name: index for index, name in enumerate(self._parameter_names)}
        if len(self._parameter_names) >= LAYER_IDENTITY_LIMIT:
            raise CaptureCapacityError("parameter catalog exceeds Int16 layer identity")
        catalog = (
            self._catalog_entries("parameter", self._parameter_names)
            + self._catalog_entries("optimizer_group", PARAM_GROUPS)
            + self._catalog_entries("learning_rate_group", self._lr_group_names)
        )
        self._prepare_activation_monitor(state)
        catalog.extend(self._catalog_entries("activation", self._activation_catalog))
        gathered = [None] * dist.get_world_size()
        torch.distributed.all_gather_object(gathered, catalog)
        if any(other != catalog for other in gathered):
            raise ValueError("rank-local monitor catalogs differ across ranks")
        self._catalog = catalog

    def _prepare_optimizer_monitors(self, state: State) -> None:
        self._optimizer_health_monitor = self._monitor_callback(state, OptimizerHealthMonitor)
        self._update_weight_monitor = self._monitor_callback(state, UpdateWeightRatioMonitor)
        self._topk_monitor = self._monitor_callback(state, TopkAccuracyMonitor)
        # The monitor disables itself without materialized logits (Liger loss, TP>1); then there are no counts.
        if self._topk_monitor._enabled and self._topk_monitor.k_values != list(_TOPK_K_VALUES):
            raise ValueError("rank-local top-k capture requires k_values=[1, 5, 20]")
        self._optimizer_monitor = self._monitor_callback(state, OptimizerMonitor)
        self._training_monitor = self._monitor_callback(state, TrainingMetricsMonitor)
        if self._optimizer_monitor.log_pre_clip_gradients:
            raise ValueError("rank-local optimizer metrics require a separate contract for pre-clip logging")
        self._update_parameter_names = list(self._update_weight_monitor._fqn_order)
        self._optimizer_parameter_names = sorted(set(self._update_parameter_names) | {
            name for name, parameter in state.model.named_parameters() if parameter.requires_grad
        })
        self._optimizer_keys = tuple(
            tuple(f"{prefix}/{name}" for prefix in _OPTIMIZER_METRIC_PREFIXES)
            for name in self._optimizer_parameter_names
        )

    def _prepare_activation_monitor(self, state: State) -> None:
        activation_monitor = self._monitor_callback(state, ActivationMonitor)
        # The capture reduces amax and L2 itself; the monitor must log them too, other statistics are its own.
        if not {"amax", "l2norm"} <= set(activation_monitor.stat_list):
            raise ValueError("rank-local activation capture requires a stat_list with amax and l2norm")
        original_names = {module: name for name, module in state.model.named_modules()}
        self._activation_modules = [
            (module, original_names[module]) for module in activation_monitor.module_names
        ]
        self._log_output = activation_monitor.log_module_output
        self._log_gradients = activation_monitor.log_gradients
        self._log_grad_input = activation_monitor.log_grad_input
        self._activation_catalog = sorted(set(self._monitored_activation_names()))
        if len(self._activation_catalog) >= LAYER_IDENTITY_LIMIT:
            raise CaptureCapacityError("activation catalog exceeds Int16 layer identity")

    def _monitored_activation_names(self) -> list[str]:
        names = []
        for module, name in self._activation_modules:
            if not isinstance(module, torch.nn.Embedding):
                names.append(f"{name}:input.0")
                if self._log_gradients and self._log_grad_input:
                    names.append(f"{name}:grad_input.0")
            output_paths = ("1", "2") if isinstance(module, MoEGate) else ("0",)
            for path in output_paths:
                if self._log_output:
                    names.append(f"{name}:output.{path}")
                if self._log_gradients:
                    names.append(f"{name}:grad_output.{path}")
        return names

    @staticmethod
    def _catalog_entries(kind: str, names: list[str]) -> list[dict[str, int | str]]:
        return [{"rank": _ALL_RANKS, "kind": kind, "index": index, "name": name} for index, name in enumerate(names)]

    def _monitor_callback(self, state: State, callback_type: type[Callback]) -> Callback:
        # Monitors set up their state in fit_start and call observers from batch_end, so they must
        # run before the sampler closes the step.
        for callback in state.callbacks[:state.callbacks.index(self._sampler)]:
            if isinstance(callback, callback_type):
                return callback
        raise ValueError(
            f"rank_local_monitors needs a {callback_type.__name__} callback listed before metric_sampler; "
            "merge the callbacks of metric-sampler-production.yaml and keep metric_sampler last"
        )

    def _prepare_monitor_additions(self, state: State) -> None:
        self._lr_monitor = self._monitor_callback(state, LRMonitor)
        self._memory_monitor = self._monitor_callback(state, MemoryMonitor)
        self._speed_monitor = self._monitor_callback(state, SpeedMonitor)
        self._stage_time_monitor = self._monitor_callback(state, StageTimeProfiler)
        self._weight_monitor = self._monitor_callback(state, WeightMonitor)
        if self._memory_monitor.memory_keys is not None:
            raise ValueError("rank-local memory capture requires the default MemoryMonitor key contract")
        if not {"amax", "l2norm"} <= set(self._weight_monitor.stat_list):
            raise ValueError("rank-local materialized weights require a stat_list with amax and l2norm")
        self._lr_group_names = sorted(
            self._lr_group_name(index, optimizer.__class__.__name__, group_index)
            for index, optimizer in enumerate(state.optimizers)
            for group_index in range(len(optimizer.param_groups))
        )
        self._lr_group_indices = {name: index for index, name in enumerate(self._lr_group_names)}
        if len(self._lr_group_names) >= LAYER_IDENTITY_LIMIT:
            raise CaptureCapacityError("learning-rate group catalog exceeds Int16 layer identity")
        if state.gigafsdp_enabled:
            root_name = self._weight_monitor.module_names[self._sampler.causal_lm]
            logical_names = [
                f"{root_name}.{name}" if root_name else name
                for name in state.model.model._meta_info["param_name_to_layer_name"]
            ]
        else:
            logical_names = [name for name, _ in state.model.named_parameters()]
        modules = {name: module for module, name in self._weight_monitor.module_names.items()}
        names = set()
        for logical_name in logical_names:
            name = cleanup_metric_name(logical_name)
            if is_internal_parameter_name(name) or not self._weight_monitor.filter_modules(name):
                continue
            if not self._weight_monitor._is_expert_name(name):
                names.add(name)
                continue
            module_name = name.rsplit(".", 1)[0] if "." in name else ""
            module = modules[module_name]
            local_experts = self._weight_monitor._get_num_local_experts(module)
            # The original WeightMonitor has no per-expert view for modules without this axis.
            if local_experts is None:
                continue
            for expert in range(local_experts * (dist.get_ep_group_size() or 1)):
                names.add(f"{name}__expert_idx_{expert}")
        self._materialized_parameter_names = sorted(names)
        self._materialized_units = self._super_tensor_units(state.model) if state.gigafsdp_enabled else {}

    @staticmethod
    def _super_tensor_units(model: torch.nn.Module) -> dict[torch.nn.Module, torch.nn.Module]:
        """The SuperTensorModule each module inside one belongs to, except the SuperTensorModule itself:
        its weight monitor hook runs after it has handed its weight buffer on."""
        # giga_fsdp is installed only where GigaFSDP trains.
        from giga_fsdp import SuperTensorModule

        units = {}
        for module in model.modules():
            if isinstance(module, SuperTensorModule):
                units.update(dict.fromkeys(module.root_module.modules(), module))
        return units

    def install(self, state: State, attempt: str) -> None:
        self._trainer_state = state
        self._install_weight_observer()
        experts = self._routed_gates[0][1].n_routed_experts
        local_experts = experts // (dist.get_ep_group_size() or 1)
        offset = (dist.get_ep_group_rank() or 0) * local_experts
        self.expert_balance_stats = ExpertBalanceStats(self._recorder, experts, local_experts, offset)
        self._install_expert_observers()
        ownership = [
            {"rank": dist.get_global_rank(), "layer": layer,
             "global_expert_ids": list(range(offset, offset + local_experts))}
            for layer, _ in self._expert_modules
        ]
        all_ownership = [None] * dist.get_world_size()
        torch.distributed.all_gather_object(all_ownership, ownership)
        self._install_router_margin_hooks(state)
        self.monitor_activations = MonitorActivationStats(
            self._recorder, self._activation_catalog, self._log_output, self._log_gradients,
            self._log_grad_input, Float8BlockwiseQTensor,
        )
        for module, name in self._activation_modules:
            self._handles.append(module.register_forward_hook(self._sampler.monitor_hook(name)))
        if dist.get_global_rank() == 0:
            self._write_monitor_files(attempt, all_ownership)
        torch.distributed.barrier()

    def _install_weight_observer(self) -> None:
        self.materialized_weights = MaterializedWeights(
            self._recorder, self._weight_monitor, self._materialized_parameter_names, self._materialized_units,
            self._parameter_rows,
        )
        native_weights = self.materialized_weights.native_observer(self._sampler)
        if native_weights is not None:  # pragma: no cover
            self._weight_observer = native_weights
        for super_tensor_module in dict.fromkeys(self._materialized_units.values()):
            self._handles.append(
                super_tensor_module.register_forward_hook(self._record_super_tensor_weights, prepend=True)
            )

    def _install_expert_observers(self) -> None:
        native = self.expert_balance_stats.native_calls(self._sampler)
        if native is not None:  # pragma: no cover
            # The extension's observers take a call without a Python call; the guard would re-raise each of their
            # failures, as it does every failure with rank-local monitors.
            self._balance_observers = [(gate, native.router(layer)) for layer, gate in self._routed_gates]
        else:
            self._balance_observers = [
                (gate, self._sampler.layer_observer("balance_observer", layer, self._observe_router_balance))
                for layer, gate in self._routed_gates
            ]
        native_experts = self.expert_balance_stats.native_expert_calls(self._sampler)
        if native_experts is not None:  # pragma: no cover
            self._expert_activation_observers = [
                (module, native_experts.experts(layer)) for layer, module in self._expert_modules
            ]
        else:
            self._expert_activation_observers = [
                (module, self._sampler.layer_observer(
                    "expert_activation_observer", layer, self._observe_expert_activations))
                for layer, module in self._expert_modules
            ]

    def _install_router_margin_hooks(self, state: State) -> None:
        margin_monitor = self._monitor_callback(state, RouterMarginMonitor)
        native_margins = self._native_router_margins()
        for layer, gate in self._routed_gates:
            if native_margins is None:
                hook = functools.partial(self._observe_router_margin, layer, gate, margin_monitor.near_m)
            else:  # pragma: no cover
                hook = native_margins.router(
                    layer, gate.scoring_func, gate.top_k, margin_monitor.near_m, gate.n_routed_experts,
                    functools.partial(self._record_router_margin, layer, gate, margin_monitor.near_m),
                )
            self._handles.append(gate.gate.register_forward_hook(hook))

    def _write_monitor_files(self, attempt: str, all_ownership: list[list[dict]]) -> None:
        directory = self._sampler.output_dir / f"attempt={attempt}"
        directory.mkdir(parents=True, exist_ok=True)
        self._write_atomically(
            directory / "monitor_parameters.json", json.dumps(self._catalog, ensure_ascii=False, indent=2)
        )
        ownership = [entry for rank_entries in all_ownership for entry in rank_entries]
        self._write_atomically(directory / "monitor_expert_ownership.json", json.dumps(ownership, indent=2))
        self._write_atomically(
            directory / "monitor_capture_contract.json",
            json.dumps(self._capture_contract(), ensure_ascii=False, indent=2),
        )

    @staticmethod
    def _write_atomically(target: Path, text: str) -> None:
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)

    def _capture_contract(self) -> dict[str, object]:
        return {
            "update_weight_parameter_names": self._update_parameter_names,
            "optimizer_parameter_names": self._optimizer_parameter_names,
            "activation_names": self._activation_catalog,
            "optimizer_group_names": PARAM_GROUPS,
            "optimizer_health_field_order": _OPTIMIZER_HEALTH_FIELD_ORDER,
            "optimizer_local_field_order": [
                "grad_l2", "moment_l2", "moment_sq_l2", "param_l2", "update_l2", "moment_bc_l2", "moment_sq_bc_l2",
                "moment_amax", "moment_sq_amax",
            ],
            "routed_layers": [layer for layer, _ in self._routed_gates],
            "global_expert_ids": list(range(self._routed_gates[0][1].n_routed_experts)),
            "expert_ownership_file": "monitor_expert_ownership.json",
            "expert_moments": (
                "Float64 sumsq and exact UInt32 token counts before accumulation/reduction; global-expert arrays "
                "contain structural zeros outside each rank's owned slice"
            ),
            "router_balance_order": _ROUTER_BALANCE_ORDER,
            "router_balance_counts_order": _ROUTER_BALANCE_COUNTS_ORDER,
            "router_balance_scope": (
                "First three sums include valid rows; biased unnormalized entropy includes every row. All calls in "
                "each rank/layer/microbatch accumulate; zero valid-token denominator means undefined (NaN) derived "
                "mean. Router z/seq/alpha means divide by forward_calls."
            ),
            "activation_statistics": (
                "amax and row-weighted mean last-axis L2 over every observed call; float32 input arithmetic, "
                "float64 row-norm sums"
            ),
            "activation_presence": "observation_count,row_count; zero means absent, paired values NaN",
            "weight_statistics": (
                "owned local logical-FQN shard after optimizer update; empty ownership is NaN with ID142 numel=0"
            ),
            "optimizer_statistics": (
                "local owned optimizer statistics before metric reductions; training gradients may already be reduced"
            ),
            "training_counter_order": _TRAINING_COUNTER_ORDER,
            "learning_rate_group_names": self._lr_group_names,
            "lr_source_instant": (
                "LRMonitor batch_end after scheduler step; all optimizer/group values, Float64 bits"
            ),
            "memory_stat_order": list(self._memory_stat_names()),
            "memory_source_instant": (
                "MemoryMonitor after_train_batch; exact memory_stats, device_free and device_total before GiB rounding"
            ),
            "training_local_order": _TRAINING_LOCAL_ORDER,
            "training_local_source": (
                "Trainer training-loop operands before rank sum and rank0 time broadcast; FLOPs before SpeedMonitor "
                "float32 cast and sum"
            ),
            "stage_time_order": list(self._stage_time_names()),
            "stage_time_source": (
                "StageTimeProfiler time.time host wallclock, complete batch-stage sums over all microbatches; not "
                "CUDA durations"
            ),
            "materialized_parameter_names": self._materialized_parameter_names,
            "materialized_weight_source": (
                "First forward materialization per FQN/expert per step before optimizer update, including existing "
                "FSDP all-gather provenance; weights constant within step. Not owned shards or "
                "pre-training-communication tensors."
            ),
            "materialized_weight_presence": "amax,wholeTensorL2,presence; NaN/NaN/0 means no materialization observed",
            "float64_and_uint64_words": (
                "little-endian UInt32 low word then high word; never numerically aggregate words"
            ),
        }

    def slot_requests(self, experts: int, routed_layers: dict[int, int]) -> list[tuple[str, int, int, int]]:
        """The slots of the rank-local monitors' observers, as (metric name, layer, microbatch, width)."""
        requests = [
            ("monitor/optimizer_health_local", NO_LAYER, NO_MICROBATCH, _OPTIMIZER_HEALTH_WIDTH),
            ("monitor/training_local_counts", NO_LAYER, NO_MICROBATCH,
             _VALUE_WITH_PRESENCE_WIDTH * len(_TRAINING_COUNTER_ORDER)),
            ("monitor/memory_local", NO_LAYER, NO_MICROBATCH,
             _VALUE_WITH_PRESENCE_WIDTH * len(self._memory_stat_names())),
            ("monitor/training_local", NO_LAYER, NO_MICROBATCH,
             _VALUE_WITH_PRESENCE_WIDTH * len(_TRAINING_LOCAL_ORDER)),
            ("monitor/stage_time_local", NO_LAYER, NO_MICROBATCH,
             _VALUE_WITH_PRESENCE_WIDTH * len(self._stage_time_names())),
        ]
        requests += [
            ("monitor/lr_group_local", index, NO_MICROBATCH, WORDS_PER_FLOAT64)
            for index in range(len(self._lr_group_names))
        ]
        requests += [
            ("monitor/materialized_weight_local", self._parameter_indices[name], NO_MICROBATCH,
             MATERIALIZED_WEIGHT_ROW_WIDTH)
            for name in self._materialized_parameter_names
        ]
        requests += [
            ("monitor/optimizer_local", self._parameter_indices[name], NO_MICROBATCH, _OPTIMIZER_LOCAL_WIDTH)
            for name in self._optimizer_parameter_names
        ]
        for name in self._update_parameter_names:
            index = self._parameter_indices[name]
            requests += [
                ("monitor/update_weight_local", index, NO_MICROBATCH, _UPDATE_WEIGHT_WIDTH),
                ("monitor/weight_local", index, NO_MICROBATCH, _WEIGHT_STATISTICS_WIDTH),
            ]
        for microbatch in range(self._sampler.microbatches):
            if self._topk_monitor._enabled:
                requests.append(("monitor/topk_local_counts", NO_LAYER, microbatch, _TOPK_COUNTS_WIDTH))
            for index in range(len(self._activation_catalog)):
                requests += [
                    ("monitor/activation_local", index, microbatch, _ACTIVATION_LOCAL_WIDTH),
                    ("monitor/activation_local_observations", index, microbatch, _ACTIVATION_LOCAL_WIDTH),
                ]
            for layer in routed_layers:
                requests += [
                    ("monitor/router_margin_local", layer, microbatch, ROUTER_MARGIN_WIDTH),
                    ("router/scores_mean", layer, microbatch, experts),
                    ("monitor/router_balance_local", layer, microbatch, _ROUTER_BALANCE_WIDTH),
                    ("monitor/router_balance_counts_local", layer, microbatch, len(_ROUTER_BALANCE_COUNTS_ORDER)),
                    ("monitor/router_bias_local", layer, microbatch, experts),
                ]
            for layer, _ in self._expert_modules:
                requests += [
                    ("monitor/expert_activation_sumsq_local", layer, microbatch, WORDS_PER_FLOAT64 * experts),
                    ("monitor/expert_activation_count_local", layer, microbatch, experts),
                ]
        return requests

    def attach(self, active: bool) -> None:
        """Hands the observers to the monitors, the routers and the experts for a captured step, or takes them back
        for a step that is not captured."""
        self._trainer_state.rank_local_training_observer = self._observe_training_local if active else None
        self._lr_monitor.rank_local_observer = self._observe_lr_group if active else None
        self._memory_monitor.rank_local_observer = self._observe_memory_local if active else None
        self._speed_monitor.rank_local_flops_observer = self._observe_local_flops if active else None
        self._stage_time_monitor.rank_local_observer = self._observe_stage_time if active else None
        self._weight_monitor.rank_local_observer = self._weight_observer if active else None
        for gate, observer in self._balance_observers:
            gate.balance_observer = observer if active else None
        for module, observer in self._expert_activation_observers:
            module.rank_local_activation_observer = observer if active else None
        self._optimizer_monitor.rank_local_observer = self._observe_optimizer_metrics if active else None
        self._training_monitor.rank_local_observer = self._observe_training_counts if active else None
        self._optimizer_health_monitor.rank_local_observer = self._observe_optimizer_health if active else None
        self._update_weight_monitor.rank_local_observer = self._observe_update_weight if active else None
        if self._topk_monitor._enabled:
            self._sampler.causal_lm.topk_observer = self._observe_topk if active else None

    def begin_step(self) -> None:
        if self.materialized_weights is not None:
            self.materialized_weights.begin_step()
        self._training_local = None
        self._local_flops = None

    def after_forward(self) -> None:
        self.monitor_activations.end_forward(self._microbatch)
        self._flush_router_margins()
        self.expert_balance_stats.flush()

    def after_backward(self) -> None:
        if self._activation_catalog:
            self.monitor_activations.flush(self._microbatch)

    def end_step(self) -> None:
        self._flush_monitor_additions()
        self.expert_balance_stats.require_binary_masks()

    def finish(self) -> None:
        if self.expert_balance_stats is not None:
            self.expert_balance_stats.finish()

    def observe_module(self, name: str, module, inputs, output) -> MonitorObservations | None:
        """The call of a monitored module in a training forward with grad, handed to the activation statistics: the
        MonitorObservations of the forward once they have been made, else None."""
        if not (module.training and self._sampler.forward_is_captured()):
            return None
        return self.monitor_activations.observe_module(name, self._microbatch, inputs, output)

    def module_slots(self, name: str, inputs, outputs) -> list[int]:  # pragma: no cover
        return self.monitor_activations.module_slots(name, inputs, outputs)

    def detach_expert_observers(self) -> None:
        for _, gate in self._routed_gates:
            gate.balance_observer = None
        for _, module in self._expert_modules:
            module.rank_local_activation_observer = None

    def shutdown(self) -> None:
        self.detach_expert_observers()
        self.expert_balance_stats = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._resolved_parameter_rows.clear()
        self.materialized_weights = None
        self._weight_observer = self._observe_materialized_weights
        self._materialized_units.clear()
        self.monitor_activations = None
        self._router_margin_slots.clear()
        self._optimizer_presence = None
        self._optimizer_keys = ()
        self._detach_monitors()
        self._router_margin_layers.clear()
        self._router_margin_graphs.clear()

    def _detach_monitors(self) -> None:
        if self._trainer_state is not None:
            self._trainer_state.rank_local_training_observer = None
            self._trainer_state = None
        if self._speed_monitor is not None:
            self._speed_monitor.rank_local_flops_observer = None
        for monitor in (
            self._lr_monitor, self._memory_monitor, self._stage_time_monitor, self._weight_monitor,
            self._optimizer_monitor, self._training_monitor, self._optimizer_health_monitor,
            self._update_weight_monitor,
        ):
            if monitor is not None:
                monitor.rank_local_observer = None
        if self._sampler.causal_lm is not None and self._sampler.rank_local_monitors:
            self._sampler.causal_lm.topk_observer = None

    @staticmethod
    def _memory_stat_names() -> tuple[str, ...]:
        return (
            "allocated_bytes.all.current", "active_bytes.all.current", "inactive_split_bytes.all.current",
            "reserved_bytes.all.current", "allocated_bytes.all.peak", "active_bytes.all.peak",
            "inactive_split_bytes.all.peak", "reserved_bytes.all.peak", "num_alloc_retries",
            "device_free", "device_total",
        )

    @staticmethod
    def _stage_time_names() -> tuple[str, ...]:
        return ("batch_total", "forward", "loss_calculation", "backward", "optimizer_step", "dataloader_fetch")

    @staticmethod
    def _uint64_words(value: int) -> tuple[int, int]:
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < _UINT64_LIMIT:
            raise ValueError("rank-local counter must be an unsigned 64-bit integer")
        return struct.unpack("<ii", struct.pack("<Q", value))

    @staticmethod
    def _float64_words(value: float) -> tuple[int, int]:
        if isinstance(value, int) and float(value) != value:
            raise ValueError("rank-local integer cannot be represented exactly as Float64")
        return struct.unpack("<ii", struct.pack("<d", value))

    @staticmethod
    def _lr_group_name(optimizer_index: int, optimizer_name: str, group_index: int) -> str:
        return f"optimizer_{optimizer_index}/{optimizer_name}/group_{group_index}"

    @staticmethod
    def _optional_words(value: object, to_words: Callable[..., tuple[int, int]]) -> tuple[int, int] | None:
        return None if value is None else to_words(value)

    @staticmethod
    def _words_and_presence(words_of_values: Iterable[tuple[int, int] | None]) -> list[int]:
        """The words of each value, zeros for an absent one, followed by a presence flag per value."""
        words: list[int] = []
        present: list[int] = []
        for value_words in words_of_values:
            words.extend(_ABSENT_WORDS if value_words is None else value_words)
            present.append(int(value_words is not None))
        return words + present

    def _observe_lr_group(self, optimizer_index: int, optimizer_name: str, group_index: int, value: float) -> None:
        name = self._lr_group_name(optimizer_index, optimizer_name, group_index)
        index = self._lr_group_indices[name]
        self._recorder.record_host_words("monitor/lr_group_local", index, NO_MICROBATCH, self._float64_words(value))

    def _observe_memory_local(self, values: dict[str, int], free: int, total: int) -> None:
        values = {**values, "device_free": free, "device_total": total}
        words = self._words_and_presence(
            self._optional_words(values.get(name), self._uint64_words) for name in self._memory_stat_names()
        )
        self._recorder.record_host_words("monitor/memory_local", NO_LAYER, NO_MICROBATCH, words)

    def _observe_training_local(self, samples: int, tokens: int, seconds: float) -> None:
        self._uint64_words(samples)
        self._uint64_words(tokens)
        self._training_local = (samples, tokens, seconds)

    def _observe_local_flops(self, value: float) -> None:
        self._float64_words(value)
        self._local_flops = value

    def _observe_stage_time(self, values: dict[str, float]) -> None:
        unknown = set(values) - set(self._stage_time_names())
        if unknown:
            raise ValueError(f"unregistered training stage timings: {sorted(unknown)}")
        words = self._words_and_presence(
            self._optional_words(values.get(name), self._float64_words) for name in self._stage_time_names()
        )
        self._recorder.record_host_words("monitor/stage_time_local", NO_LAYER, NO_MICROBATCH, words)

    def _observe_materialized_weights(self, module: torch.nn.Module) -> None:
        if not (module.training and self._sampler.forward_is_captured()):
            return
        self.materialized_weights.observe(module)

    def _record_super_tensor_weights(self, super_tensor_module: torch.nn.Module, inputs, output) -> None:
        # Registered before the SuperTensor's own forward hook, which hands its weight buffer on.
        if not (super_tensor_module.training and self._sampler.forward_is_captured()):
            return
        self.materialized_weights.end_super_tensor(super_tensor_module)

    def _flush_monitor_additions(self) -> None:
        samples_words = tokens_words = seconds_words = flops_words = None
        if self._training_local is not None:
            samples, tokens, seconds = self._training_local
            samples_words = self._uint64_words(samples)
            tokens_words = self._uint64_words(tokens)
            seconds_words = self._float64_words(seconds)
        if self._local_flops is not None:
            flops_words = self._float64_words(self._local_flops)
        words = self._words_and_presence((samples_words, tokens_words, seconds_words, flops_words))
        self._recorder.record_host_words("monitor/training_local", NO_LAYER, NO_MICROBATCH, words)
        self.materialized_weights.flush()

    def _parameter_rows(self, metric_name: str, names: tuple[str, ...]) -> ResolvedRows:
        # Resolved once per name list: the device index behind it is a host-to-device copy.
        rows = self._resolved_parameter_rows.get((metric_name, names))
        if rows is None:
            unknown = [name for name in names if name not in self._parameter_indices]
            if unknown:
                raise ValueError(f"unregistered {metric_name} identities: {unknown[:3]}")
            rows = self._recorder.resolve_rows(
                metric_name, [self._parameter_indices[name] for name in names], NO_MICROBATCH
            )
            self._resolved_parameter_rows[(metric_name, names)] = rows
        return rows

    def _observe_router_balance(self, layer: int, *operands) -> None:
        if self._sampler.forward_is_captured():
            self.expert_balance_stats.observe_router(layer, self._microbatch, *operands)

    def _observe_expert_activations(self, layer: int, intermediate: torch.Tensor, counts: list[int]) -> None:
        if self._sampler.forward_is_captured():
            self.expert_balance_stats.observe_experts(layer, self._microbatch, intermediate, counts)

    def _observe_training_counts(self, values: tuple) -> None:
        words = self._words_and_presence(self._optional_words(value, self._counter_words) for value in values)
        self._recorder.record_host_words("monitor/training_local_counts", NO_LAYER, NO_MICROBATCH, words)

    @staticmethod
    def _counter_words(value: object) -> tuple[int, int]:
        scalar = value.item() if isinstance(value, torch.Tensor) else value
        count = int(scalar)
        if count != scalar or not 0 <= count < _UINT64_LIMIT:
            raise ValueError("training monitor counter is not an exact nonnegative UInt64")
        return count & _UINT32_MASK, count >> _UINT32_BITS

    def _observe_optimizer_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        keys = self._optimizer_keys
        unknown = set(metrics).difference(*keys)
        if unknown:
            raise ValueError(f"optimizer monitor emitted unregistered local identities: {sorted(unknown)}")
        if not keys:
            return
        device = self._recorder.buffer.device
        pattern = tuple(key in metrics for row in keys for key in row)
        if self._optimizer_presence is None or self._optimizer_presence[0] != pattern:
            presence = torch.tensor(pattern, dtype=torch.float32).view(len(keys), -1)
            emitted = torch.tensor([index for index, present in enumerate(pattern) if present], dtype=torch.long)
            self._optimizer_presence = (pattern, presence.to(device), emitted.to(device))
        _, presence, emitted = self._optimizer_presence
        # Only the emitted metrics are stacked, most keys being absent: a stack costs host time per input.
        values = torch.full((len(pattern),), float("nan"), device=device)
        if emitted.numel():
            with torch.no_grad():
                values[emitted] = torch.stack(
                    [self._float32_scalar(metrics[key]) for row in keys for key in row if key in metrics]
                )
        payload = torch.cat((values.view(len(keys), -1), presence), dim=1)
        rows = self._parameter_rows("monitor/optimizer_local", tuple(self._optimizer_parameter_names))
        self._recorder.record_rows(rows, payload)

    @staticmethod
    def _float32_scalar(metric: torch.Tensor) -> torch.Tensor:
        # A float32 scalar is stacked as it is: each of detach, float and reshape is an op per metric.
        value = metric if metric.dtype == torch.float32 else metric.float()
        return value.reshape(()) if value.ndim else value

    def _observe_optimizer_health(self, values: torch.Tensor) -> None:
        self._recorder.record(
            "monitor/optimizer_health_local", NO_LAYER, NO_MICROBATCH, values.detach().contiguous().view(torch.int32)
        )

    def _observe_update_weight(self, names: list[str], values: torch.Tensor, weight_stats: torch.Tensor) -> None:
        if names != self._update_parameter_names:
            raise ValueError("parameter identity changed after catalog publication")
        key = tuple(names)
        self._recorder.record_rows(
            self._parameter_rows("monitor/update_weight_local", key), values.detach().contiguous().view(torch.int32)
        )
        self._recorder.record_rows(self._parameter_rows("monitor/weight_local", key), weight_stats)

    def _observe_topk(self, counts: torch.Tensor) -> None:
        self._recorder.record("monitor/topk_local_counts", NO_LAYER, self._microbatch, counts)

    def _observe_router_margin(self, layer: int, gate: MoEGate, near_count: int, module, inputs, logits) -> None:
        if not (module.training and self._sampler.forward_is_captured()):
            return
        # Computed now, so no layer's logits are kept until the end of the forward.
        if layer in self._router_margin_layers:
            raise ValueError(f"router margin of layer {layer} was observed twice in microbatch {self._microbatch}")
        self._router_margin_layers.add(layer)
        self._record_router_margin(layer, gate, near_count, logits)

    def _flush_router_margins(self) -> None:
        self._router_margin_layers.clear()

    def _native_router_margins(self) -> "_NativeRouterMargins | None":  # pragma: no cover
        """The extension's router margin hooks, which take a call as _observe_router_margin does, without a Python
        call. None without the extension or over a host recorder: the extension writes the recorder's CUDA words."""
        extension = activation_ext.prebuilt()
        if not extension or not self._recorder.buffer.is_cuda:
            return None
        return extension.RouterMarginCalls(
            self._sampler, self._recorder, self._native_router_margin_slot, self._router_margin_graph,
            self._router_margin_layers, self._recorder.recorded_flags(), torch._C._current_graph_task_id,
        )

    def _native_router_margin_slot(self, layer: int, microbatch: int) -> tuple[int, int, int]:  # pragma: no cover
        """The (offset, width, slot index) of the router margin slot of a call RouterMarginCalls takes."""
        slot = self._recorder.resolve("monitor/router_margin_local", layer, microbatch)
        return slot.offset, slot.width, slot.slot_index

    def _record_router_margin(self, layer: int, gate: MoEGate, near_count: int, logits: torch.Tensor) -> None:
        # Detached logits keep the computation out of the autograd graph without a no_grad scope.
        logits = logits.detach().reshape(-1, gate.n_routed_experts)
        if logits.shape[0] == 0:
            absent = torch.full((ROUTER_MARGIN_WIDTH,), float("nan"), device=logits.device)
            self._recorder.record("monitor/router_margin_local", layer, self._microbatch, absent)
            return
        if gate.scoring_func not in ("softmax", "sigmoid"):
            raise ValueError("rank-local router margins require softmax or sigmoid")
        if gate.n_routed_experts <= gate.top_k:
            raise ValueError("router margin has no unselected experts")
        if logits.is_cuda:  # pragma: no cover
            graphed = self._router_margin_graph(gate.scoring_func, gate.top_k, near_count, logits)
            slot = self._router_margin_slot(layer)
            self._recorder.mark_recorded([slot.slot_index])
            self._recorder.view(slot).copy_(graphed(logits))
            return
        stats = router_margin_stats(gate.scoring_func, gate.top_k, near_count, logits)
        self._recorder.record("monitor/router_margin_local", layer, self._microbatch, stats)

    def _router_margin_graph(
        self, scoring_function: str, top_k: int, near_count: int, logits: torch.Tensor
    ) -> GraphedComputation:  # pragma: no cover
        # One graph replay instead of dozens of small launches per layer.
        key = (tuple(logits.shape), logits.dtype, logits.device, scoring_function, top_k, near_count)
        graphed = self._router_margin_graphs.get(key)
        if graphed is None:
            compute = functools.partial(router_margin_graph_stats, scoring_function, top_k, near_count)
            graphed = self._router_margin_graphs[key] = GraphedComputation(compute, logits)
        return graphed

    def _router_margin_slot(self, layer: int) -> ResolvedSlot:  # pragma: no cover
        key = (layer, self._microbatch)
        slot = self._router_margin_slots.get(key)
        if slot is None:
            slot = self._router_margin_slots[key] = self._recorder.resolve(
                "monitor/router_margin_local", layer, self._microbatch
            )
        return slot
