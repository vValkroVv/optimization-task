# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor weight values during training."""

import logging
import warnings
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.loggers.wandb_logger import WandBLogger
from composer.utils import dist, global_weightlog_state

__all__ = ['WeightMonitor']

log = logging.getLogger(__name__)


class WeightMonitor(Callback):
    """Logs stats of model weights.

    This callback triggers at a user defined interval and logs statistics of
    model parameters.

    Dense (non-expert) parameters are collected from module forward hooks so
    that, under FSDP, statistics are computed when unsharded weights are already
    materialized. Expert parameters are collected from forward hooks on one
    replica of each expert-parallel group and then gathered within the EP group.

    Only floating-point / complex 2D tensors are logged.

    Args:
        interval (Union[int, str, Time], optional): Time string specifying how
            often to log weights. For example, ``interval='25ba'`` means every
            25 batches. Defaults to ``'25ba'``.
        ignore_module_types (Optional[List[str]], optional): Substrings of
            parameter names to skip. Defaults to ``None``.
        keep_only_module_types (Optional[List[str]], optional): If set, only
            parameter names containing at least one of these substrings are
            logged. Defaults to ``None``.
        expert_module_types (Optional[List[str]], optional): Substrings used to
            identify expert parameters. Expert metrics are collected on one EP
            replica (`ep_fsdp_rank == 0`) and gathered inside EP group.
            Defaults to ``['expert']``.
        only_log_wandb (bool, optional): If ``True``, logs only to Weights and
            Biases destinations when present. Defaults to ``True``.
        stat_list (Optional[List[str]], optional): List of statistics to log.
            Supported values: ``amax``, ``amin``, ``max``, ``min``, ``amean``,
            ``average``, ``l2norm``, ``kurtosis``.
            Defaults to ``['amax', 'l2norm', 'average']``.
        tp_size (int, optional): Tensor parallel size. This monitor only
            supports ``tp_size == 1``. Defaults to ``1``.
    """

    def __init__(
        self,
        interval: Union[int, str, Time] = '25ba',
        ignore_module_types: Optional[List[str]] = None,
        keep_only_module_types: Optional[List[str]] = None,
        expert_module_types: Optional[List[str]] = None,
        only_log_wandb: bool = True,
        stat_list: Optional[List[str]] = None,
        tp_size: int = 1,
    ):
        self.ignore_module_types = ignore_module_types
        self.keep_only_module_types = keep_only_module_types
        self.expert_module_types = [s.lower() for s in (expert_module_types or ['expert'])]
        self.only_log_wandb = only_log_wandb
        self.stat_list = stat_list or ['amax', 'l2norm', 'average']
        self.tp_size = tp_size

        self.interval = Time.from_input(interval, TimeUnit.BATCH)
        if self.interval.unit == TimeUnit.BATCH and self.interval < Time.from_timestring('10ba'):
            warnings.warn(
                f'Currently the WeightMonitor interval is set to {self.interval} '
                f'which is below our recommended value of 10ba. We recommend '
                f'raising the interval to at least 10ba.'
            )
        if self.interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f'Invalid time unit for parameter interval: {self.interval.unit}')

        self.last_train_time_value_logged = -1
        self.rank_local_observer = None
        self.capture_weights = False
        self.capture_dense_weights = False
        self.capture_expert_weights = False
        self.module_names: Dict[torch.nn.Module, str] = {}
        self.handles: List[Any] = []
        self.weight_metrics: Dict[str, torch.Tensor] = {}
        self.external_metrics: Dict[str, torch.Tensor] = {}

    def fit_start(self, state: State, logger: Logger):
        assert self.tp_size == 1, (
            f'WeightMonitor only supports tp_size == 1, but got tp_size={self.tp_size}. '
            'Tensor-parallel sharded weights need different metric aggregation semantics.'
        )

        if not self.module_names:
            self.create_module_names(state.model)
        global_weightlog_state.set_monitor_variable(self)
        global_weightlog_state.set_enable_monitor_variable(False)
        self.attach_hooks(state.model)
        log.info('Attached weight logging hooks on %d modules', len(self.handles))

    def _should_log(self, current_time_value: int) -> bool:
        if current_time_value == self.last_train_time_value_logged:
            return False
        return current_time_value % self.interval.value == 0

    def before_train_batch(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value
        should_log = self._should_log(current_time_value)

        self.capture_dense_weights = should_log and dist.get_global_rank() == 0
        self.capture_expert_weights = should_log and self._should_capture_expert_weights()
        self.capture_weights = self.capture_dense_weights or self.capture_expert_weights

        global_weightlog_state.set_enable_monitor_variable(self.capture_weights)

        if not self.capture_weights:
            self.weight_metrics = {}
            self.external_metrics = {}
            return

        self.weight_metrics = {}
        self.external_metrics = {}

    def batch_end(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value

        if not self.capture_weights:
            self.weight_metrics = {}
            self.external_metrics = {}
            self.capture_dense_weights = False
            self.capture_expert_weights = False
            self.capture_weights = False
            global_weightlog_state.set_enable_monitor_variable(False)
            return

        self.last_train_time_value_logged = current_time_value
        self.capture_dense_weights = False
        self.capture_expert_weights = False
        self.capture_weights = False
        global_weightlog_state.set_enable_monitor_variable(False)

        merged_metrics = self._finalize_metrics_for_logging()

        if dist.get_global_rank() == 0 and merged_metrics:
            self._log_metrics(logger, merged_metrics, state.timestamp.batch.value)

        self.weight_metrics = {}
        self.external_metrics = {}

    def fit_end(self, state: State, logger: Logger):
        global_weightlog_state.set_enable_monitor_variable(False)
        global_weightlog_state.set_monitor_variable(None)
        self.remove_hooks()

    def _log_metrics(self, logger: Logger, metrics: Dict[str, float], step: int):
        if self.only_log_wandb:
            wandb_loggers = [ld for ld in logger.destinations if isinstance(ld, WandBLogger)]
            if wandb_loggers:
                for wandb_logger in wandb_loggers:
                    wandb_logger.log_metrics(metrics, step)
            else:
                logger.log_metrics(metrics)
        else:
            logger.log_metrics(metrics)

    def _finalize_metrics_for_logging(self) -> Dict[str, float]:
        weight_metrics = self._gather_weight_metrics(self.weight_metrics)
        external_metrics = self._tensor_metrics_to_float(self.external_metrics)

        if dist.get_global_rank() != 0:
            return {}

        merged_metrics = {}
        merged_metrics.update(weight_metrics)
        merged_metrics.update(external_metrics)
        return merged_metrics

    def _tensor_metrics_to_float(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {k: v.item() for k, v in metrics.items()}

    def _gather_weight_metrics(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
        local_metrics = self._tensor_metrics_to_float(metrics)

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return local_metrics

        ep_group = dist.get_ep_group()

        if dist.is_trivial_process_group(ep_group):
            return local_metrics

        ep_group_size = dist.get_ep_group_size()
        gathered_metrics: List[Optional[Dict[str, float]]] = [None for _ in range(ep_group_size)]
        torch.distributed.all_gather_object(gathered_metrics, local_metrics, group=ep_group)
        if dist.get_ep_group_rank() != 0:
            return {}

        if dist.get_global_rank() != 0:
            return {}

        merged_metrics: Dict[str, float] = {}
        for rank_metrics in gathered_metrics:
            if rank_metrics:
                for metric_name, metric_value in rank_metrics.items():
                    # Keep first key occurrence if metrics collide across EP ranks.
                    merged_metrics.setdefault(metric_name, metric_value)
        return merged_metrics

    def filter_modules(self, module_name: str) -> bool:
        if self.ignore_module_types is not None:
            for ignore_module_type in self.ignore_module_types:
                if ignore_module_type in module_name:
                    return False

        if self.keep_only_module_types is not None:
            for keep_only_module in self.keep_only_module_types:
                if keep_only_module in module_name:
                    return True
            return False
        return True

    def attach_hooks(self, model: torch.nn.Module):
        model.apply(self._register_hook)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _register_hook(self, module: torch.nn.Module):
        self.handles.append(module.register_forward_hook(self.base_hook))

    def base_hook(self, module: torch.nn.Module, input: Tuple[Any, ...], output: Any):
        if self.rank_local_observer is not None:
            self.rank_local_observer(module)
        if not self.capture_weights:
            return

        module_name = self.module_names.get(module, '')
        for param_name, param in self._iter_module_parameters(module, include_tensor_views=True):
            if is_internal_parameter_name(param_name):
                continue

            full_name = param_name if module_name == '' else f'{module_name}.{param_name}'
            cleaned_name = cleanup_metric_name(full_name)

            if not cleaned_name or is_internal_parameter_name(cleaned_name):
                continue
            is_expert_param = self._is_expert_name(cleaned_name)

            if is_expert_param:
                if not self.capture_expert_weights:
                    continue
                if not self.filter_modules(cleaned_name):
                    continue

                for expert_idx, expert_param in self._iter_expert_param_views(module, cleaned_name, param.detach()):
                    expert_name = f'{cleaned_name}__expert_idx_{expert_idx}'
                    if self._has_metrics_for_param(expert_name, self.weight_metrics, metric_prefix='weights'):
                        continue
                    self.add_metrics(
                        self.weight_metrics,
                        expert_name,
                        expert_param,
                        metric_prefix='weights',
                    )
                continue

            if not self.capture_dense_weights:
                continue
            if not self.filter_modules(cleaned_name):
                continue
            if self._has_metrics_for_param(cleaned_name, self.weight_metrics, metric_prefix='weights'):
                continue

            self.add_metrics(self.weight_metrics, cleaned_name, param.detach(), metric_prefix='weights')

    def _has_metrics_for_param(self, param_name: str, metrics: Dict[str, torch.Tensor], metric_prefix: str):
        if not self.stat_list:
            return False
        metric_name = f'{metric_prefix}_{self.stat_list[0]}/{param_name}'
        return metric_name in metrics

    def _iter_expert_param_views(
        self,
        module: torch.nn.Module,
        cleaned_name: str,
        param: torch.Tensor,
    ) -> List[Tuple[int, torch.Tensor]]:
        stack = self._expert_param_stack(module, cleaned_name, param)
        if stack is None:
            return []
        expert_index_offset, experts = stack
        return [(expert_index_offset + expert_idx, experts[expert_idx]) for expert_idx in range(experts.shape[0])]

    def _expert_param_stack(
        self,
        module: torch.nn.Module,
        cleaned_name: str,
        param: torch.Tensor,
    ) -> Optional[Tuple[int, torch.Tensor]]:
        """The local experts' slices of ``param`` as one view with a leading expert axis, and the
        global index of the first expert; None if ``param`` has no per-expert layout."""
        num_local_experts = self._get_num_local_experts(module)
        if num_local_experts is None or num_local_experts <= 0:
            return None
        expert_index_offset = self._get_expert_index_offset(num_local_experts)

        if param.ndim >= 1 and param.shape[0] == num_local_experts:
            return expert_index_offset, param

        if cleaned_name.endswith('experts.weight1') and param.ndim == 2:
            if param.shape[1] % num_local_experts == 0:
                return expert_index_offset, param.reshape(num_local_experts, param.shape[0], -1)

        if cleaned_name.endswith('experts.weight2') and param.ndim == 2:
            if param.shape[0] % num_local_experts == 0:
                return expert_index_offset, param.reshape(num_local_experts, -1, param.shape[1])

        if num_local_experts == 1:
            # One-expert-per-rank configuration should still emit expert metrics.
            return expert_index_offset, param.unsqueeze(0)

        return None

    def _get_num_local_experts(self, module: torch.nn.Module) -> Optional[int]:
        num_local_experts = getattr(module, 'num_local_experts', None)
        return num_local_experts if isinstance(num_local_experts, int) and num_local_experts > 0 else None

    def _iter_module_parameters(
        self,
        module: torch.nn.Module,
        include_tensor_views: bool,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        seen_parameter_names: Set[str] = set()
        for param_name, param in module.named_parameters(recurse=False):
            seen_parameter_names.add(param_name)
            yield param_name, param

        if not include_tensor_views:
            return

        # Fallback for modules that expose unsharded parameter views as tensors
        # during forward (for example inside FSDP all-gathered regions).
        for tensor_name, value in vars(module).items():
            if tensor_name in seen_parameter_names:
                continue
            if is_internal_parameter_name(tensor_name):
                continue
            if isinstance(value, torch.Tensor) and value.requires_grad:
                yield tensor_name, value

    def create_module_names(self, model: torch.nn.Module):
        self.module_names = {module: cleanup_metric_name(name) for name, module in model.named_modules()}

    def add_external_metric(self, name: str, value: torch.Tensor, metric_prefix: str, suffix: str = ""):
        if not self.capture_weights:
            return
        self.add_metrics(self.external_metrics, f"{name}{suffix}", value, metric_prefix=metric_prefix)

    def add_metrics(
        self,
        metrics: Dict[str, torch.Tensor],
        name: str,
        value: torch.Tensor,
        metric_prefix: str = 'weights',
    ):
        if value.dtype == torch.bool:
            return
        if not (value.is_floating_point() or value.is_complex()):
            return

        with torch.no_grad():
            for stat_name in self.stat_list:
                stat_value = compute_weight_stats(value, stat_name)
                metric_name = f'{metric_prefix}_{stat_name}/{name}'
                metrics[metric_name] = stat_value

    def _is_expert_name(self, cleaned_name: str) -> bool:
        lowered_name = cleaned_name.lower()
        return any(expert_key in lowered_name for expert_key in self.expert_module_types)

    def _get_expert_index_offset(self, num_local_experts: int) -> int:
        ep_group_rank = dist.get_ep_group_rank()
        if ep_group_rank is None:
            return 0
        return ep_group_rank * num_local_experts

    def _should_capture_expert_weights(self) -> bool:
        if dist.is_trivial_process_group(dist.get_ep_group()):
            return dist.get_global_rank() == 0
        ep_fsdp_group_rank = dist.get_ep_fsdp_group_rank()
        return ep_fsdp_group_rank is None or ep_fsdp_group_rank == 0


def cleanup_metric_name(metric_name: str):
    metric_name = metric_name.replace("._module", "")
    metric_name = metric_name.replace(".root_module", "")
    metric_name = metric_name.replace("._orig_mod", "")
    metric_name = metric_name.replace("._checkpoint_wrapped_module", "")
    metric_name = metric_name.replace("._tp_linear_submodule", "")
    metric_name = metric_name.replace("._fsdp_wrapped_module", "")
    metric_name = metric_name.replace("._forward_module", "")
    return metric_name.strip(".")


def is_internal_parameter_name(name: str) -> bool:
    leaf_name = name.rsplit(".", 1)[-1]
    is_gigafsdp_flatten_name = (
        leaf_name == 'flatten'
        or leaf_name == 'flatten_freezed'
        or leaf_name.startswith('flatten_')
        or leaf_name.startswith('flatten_freezed_')
    )
    return (
        name.startswith("_")
        or "._flat_param" in name
        or name.endswith("._flat_param")
        or ".flat_param" in name
        or "flat_param" == name
        or is_gigafsdp_flatten_name
    )

def compute_weight_stats(value: torch.Tensor, stat_name: str):
    if stat_name == "amax":
        output = value.abs().amax()
    elif stat_name == "amin":
        output = value.abs().amin()
    elif stat_name == "max":
        output = value.amax()
    elif stat_name == "min":
        output = value.amin()
    elif stat_name == "amean":
        output = value.abs().mean()
    elif stat_name == "l2norm":
        output = torch.linalg.vector_norm(value)
    elif stat_name == "average":
        output = value.mean()
    elif stat_name == "kurtosis":
        output = compute_kurtosis(value)
    else:
        raise ValueError(f'This statistic {stat_name} is not supported!')
    return output.view(1)


def compute_kurtosis(value: torch.Tensor):
    mean = torch.mean(value, dim=-1).unsqueeze(-1)
    diffs = value - mean
    m_4 = torch.mean(torch.pow(diffs, 4), dim=-1)
    var = torch.mean(torch.pow(diffs, 2), dim=-1)
    return (m_4 / (var**2)).mean()
