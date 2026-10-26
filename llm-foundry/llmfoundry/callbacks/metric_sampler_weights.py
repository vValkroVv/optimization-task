from __future__ import annotations

from typing import Callable

import torch
from composer.callbacks.weight_monitor import WeightMonitor, cleanup_metric_name, is_internal_parameter_name

from metric_sampler.capture.kernels import activation_ext
from metric_sampler.capture.recorder import RankRecorder, ResolvedRows
from metric_sampler.capture.stats.helpers.norms import record_weight_norm_rows

# A weight's row: its absolute max, its L2 norm and a presence flag.
MATERIALIZED_WEIGHT_ROW_WIDTH = 3
_PRESENCE_COLUMN = MATERIALIZED_WEIGHT_ROW_WIDTH - 1

# Weights recorded in one extension call, per launch the weights would have been reduced in alone: their names,
# the tensors that hold them with the expert count of each (0 for one weight, else a stack of the experts' weights)
# and the element count of the launch's largest weight.
_WeightLaunches = list[tuple[list[str], list[torch.Tensor], list[int], int]]


class MaterializedWeights:
    """Captures the weights the weight monitor records as the training forward materializes them: each weight once
    per step, at its first forward, before the optimizer updates it, as a `monitor/materialized_weight_local` row.
    `names` are their logical FQNs (an expert's with its global expert index), whose rows `parameter_rows`
    resolves. `units` maps each module inside a SuperTensorModule to it: the weights are valid only while its
    forward runs, so those of the modules that ran are recorded together when it ends."""

    def __init__(
        self,
        recorder: RankRecorder,
        monitor: WeightMonitor,
        names: list[str],
        units: dict[torch.nn.Module, torch.nn.Module],
        parameter_rows: Callable[[str, tuple[str, ...]], ResolvedRows],
    ) -> None:
        self._recorder = recorder
        self._monitor = monitor
        self._names = names
        self._name_set = frozenset(names)
        self._units = units
        self._parameter_rows = parameter_rows
        self._seen: set[str] = set()
        # Modules whose every weight is recorded this step: weights are constant within a step, so
        # their later forwards are skipped before any work.
        self._modules_seen: set[torch.nn.Module] = set()
        # Per module: its recorded weights as (attribute, metric name, holds experts), from the first
        # forward that captured it; a module without any is skipped from then on.
        self._plans: dict[torch.nn.Module, list[tuple[str, str, bool]]] = {}
        # Per SuperTensorModule, the modules of it that ran this step and are not recorded yet.
        self._pending: dict[torch.nn.Module, dict[torch.nn.Module, None]] = {}
        self._expert_names: dict[tuple[str, int, int], tuple[str, ...]] = {}

    def begin_step(self) -> None:
        self._seen.clear()
        self._modules_seen.clear()
        self._pending.clear()

    def observe(self, module: torch.nn.Module) -> None:
        """The weight monitor's call for `module` in a training forward."""
        if module in self._modules_seen:
            return
        super_tensor_module = self._units.get(module)
        if super_tensor_module is None:
            self._record_modules((module,))
        else:
            self._pending.setdefault(super_tensor_module, {})[module] = None

    def native_observer(self, sampler: object) -> Callable[[torch.nn.Module], None] | None:  # pragma: no cover
        """The extension's observer of the weight monitor's calls, which takes a call as the monitor capture's
        _observe_materialized_weights and observe do, without a Python call: only for a module in training, in a
        training forward with grad, as `sampler`'s _capture_forward_active and _grad_forward tell. None without the
        extension or over a host recorder, as for the other observers of the extension."""
        extension = activation_ext.prebuilt()
        if not extension or not self._recorder.buffer.is_cuda:
            return None
        return extension.MaterializedWeightObserver(
            sampler, self._modules_seen, self._units, self._pending, self._record_modules,
            torch._C._current_graph_task_id,
        )

    def end_super_tensor(self, super_tensor_module: torch.nn.Module) -> None:
        """Records the modules of `super_tensor_module` that ran, before it hands its weight buffer on."""
        pending = self._pending.pop(super_tensor_module, None)
        if pending:
            self._record_modules(tuple(pending))

    def flush(self) -> None:
        """Records the weights no forward materialized this step as absent: NaN statistics, presence 0."""
        unseen = tuple(name for name in self._names if name not in self._seen)
        if unseen:
            rows = self._parameter_rows("monitor/materialized_weight_local", unseen)
            absent = torch.full(
                (len(unseen), MATERIALIZED_WEIGHT_ROW_WIDTH), float("nan"), device=self._recorder.buffer.device
            )
            absent[:, _PRESENCE_COLUMN] = 0.0
            self._recorder.record_rows(rows, absent)

    def _record_modules(self, modules: tuple[torch.nn.Module, ...]) -> None:
        """Records the weights of `modules` in one extension call, each module's weights reduced as in
        a call of their own."""
        launches: _WeightLaunches = []
        # One no_grad scope instead of a detach per weight.
        with torch.no_grad():
            recorded = [module for module in modules if self._add_module_weights(module, launches)]
            if launches:
                self._record_launches(launches)
        self._modules_seen.update(recorded)

    def _add_module_weights(self, module: torch.nn.Module, launches: _WeightLaunches) -> bool:
        """Adds the launches that record the weights of `module` not recorded yet this step; False when
        one of them is left unrecorded."""
        plan = self._plans.get(module)
        if plan is None:
            plan = self._plans[module] = self._plan(module)
        if not plan:
            return True
        tensors = [self._module_tensor(module, attribute) for attribute, _, _ in plan]
        if any(tensor is None for tensor in tensors):
            # A weight view of the first capture is absent now: plan this module again.
            plan = self._plans[module] = self._plan(module)
            tensors = [self._module_tensor(module, attribute) for attribute, _, _ in plan]
        names, values = [], []
        recorded = True
        for (_, name, holds_experts), tensor in zip(plan, tensors):
            if holds_experts:
                stack = self._monitor._expert_param_stack(module, name, tensor)
                recorded = stack is not None and self._add_expert_weights(name, *stack, launches) and recorded
            else:
                names.append(name)
                values.append(tensor)
        if names:
            recorded = self._add_dense_weights(names, values, launches) and recorded
        return recorded

    def _plan(self, module: torch.nn.Module) -> list[tuple[str, str, bool]]:
        """The weights the weight monitor records for `module`, with the same filters it applies."""
        monitor = self._monitor
        module_name = monitor.module_names[module]
        plan = []
        for parameter_name, value in monitor._iter_module_parameters(module, include_tensor_views=True):
            if is_internal_parameter_name(parameter_name):
                continue
            name = cleanup_metric_name(f"{module_name}.{parameter_name}" if module_name else parameter_name)
            if not name or is_internal_parameter_name(name) or not monitor.filter_modules(name):
                continue
            if not (value.is_floating_point() or value.is_complex()):
                continue
            plan.append((parameter_name, name, monitor._is_expert_name(name)))
        return plan

    @staticmethod
    def _module_tensor(module: torch.nn.Module, attribute: str) -> torch.Tensor | None:
        # A registered parameter, or a tensor attribute such as a gathered view of one.
        parameter = module._parameters.get(attribute)
        if parameter is not None:
            return parameter
        value = vars(module).get(attribute)
        return value if isinstance(value, torch.Tensor) else None

    def _add_expert_weights(
        self, name: str, first_expert: int, experts: torch.Tensor, launches: _WeightLaunches
    ) -> bool:
        """Adds the launch that records every local expert of one weight, unless already recorded this
        step; False when there is nothing to record."""
        key = (name, first_expert, experts.shape[0])
        expert_names = self._expert_names.get(key)
        if expert_names is None:
            expert_names = tuple(f"{name}__expert_idx_{first_expert + index}" for index in range(experts.shape[0]))
            self._require_registered(expert_names)
            self._expert_names[key] = expert_names
        if expert_names[0] in self._seen:
            return True
        if experts.numel() == 0:
            return False
        launches.append((list(expert_names), [experts], [experts.shape[0]], experts.numel() // experts.shape[0]))
        return True

    def _require_registered(self, names) -> None:
        unknown = [name for name in names if name not in self._name_set]
        if unknown:
            raise ValueError(f"unregistered materialized weight: {unknown[0]}")

    def _add_dense_weights(
        self, names: list[str], values: list[torch.Tensor], launches: _WeightLaunches
    ) -> bool:
        """Adds the launch that records the weights not yet recorded this step; False when an empty one
        is left unrecorded."""
        self._require_registered(names)
        unseen = [index for index, name in enumerate(names) if name not in self._seen]
        kept = [index for index in unseen if values[index].numel() > 0]
        if not kept:
            return not unseen
        if len(kept) < len(names):
            names = [names[index] for index in kept]
            values = [values[index] for index in kept]
        launches.append((names, values, [0] * len(values), max(value.numel() for value in values)))
        return len(kept) == len(unseen)

    def _record_launches(self, launches: _WeightLaunches) -> None:
        names = [name for launch_names, _, _, _ in launches for name in launch_names]
        values = [value for _, launch_values, _, _ in launches for value in launch_values]
        experts = [count for _, _, launch_experts, _ in launches for count in launch_experts]
        # A weight's sums are ordered by its launch's largest weight: the order they have when the
        # launch is recorded alone.
        launch_numels = [numel for launch_names, _, _, numel in launches for _ in launch_names]
        # Each norm keeps the weight's dtype, as the weight monitor's statistics do.
        record_weight_norm_rows(
            self._recorder, self._parameter_rows("monitor/materialized_weight_local", tuple(names)), values, experts,
            launch_numels,
        )
        self._seen.update(names)
