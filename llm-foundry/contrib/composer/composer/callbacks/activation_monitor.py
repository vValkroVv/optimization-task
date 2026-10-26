# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor activation values during training."""

import gc
import logging
import re
import warnings
from collections import defaultdict
from functools import partial
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.loggers.wandb_logger import WandBLogger
from composer.utils import TimerType, callback_timer, dist, ensure_tuple, global_actlog_state

log = logging.getLogger(__name__)
__all__ = ['ActivationMonitor']

global_actlog_state.set_enable_monitor_variable(False)

def replace_layer_pattern(input_string):
    # Use re.sub to replace the first occurrence of .\d+. with .{}.
    return re.sub(r'\.\d+\.', '.{}.', input_string, count=1)

@torch._dynamo.allow_in_graph
class BackwardLoggingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args): # noop
        ctx.module_name, ctx.grad_type, ctx.layer_depends = args[-3:]
        args = args[:-3]
        return args

    @staticmethod
    def backward(ctx, *grad_output):
        for i, tensor in enumerate(grad_output):
            if not isinstance(tensor, torch.Tensor):
                continue
            global_actlog_state.use_monitor_variable(tensor,
                                                     ctx.module_name, 
                                                     f"{ctx.grad_type}.{i}",
                                                     layer_depends=ctx.layer_depends,
                                                     is_forward=False)

        return grad_output + (None, None, None)

def log_backward(input_tensor : Union[Tuple | torch.Tensor], 
                 module_name : str, 
                 grad_type: str,
                 layer_depends: bool) -> Union[Tuple | torch.Tensor]:
    input_tensor = ensure_tuple(input_tensor)
    output_tensor =  BackwardLoggingFunction.apply(*input_tensor, 
                                                    module_name, 
                                                    grad_type,
                                                    layer_depends)
            
    if isinstance(output_tensor, tuple) and len(output_tensor) == 1:
        output_tensor, = output_tensor
    
    return output_tensor


class ActivationMonitor(Callback):
    """Logs stats of activation inputs and outputs.

    This callback triggers at a user defined interval, and logs some simple statistics of the inputs, outputs for every
    torch module. This is done by attaching a forward hook to the module. Additionally, when after we finish logging
    we detach the forwards hook.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import ActivationMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[ActivationMonitor()],
            ... )

    The metrics are logged by the :class:`.Logger` to the following keys described below. Over an input of shape
    (batch, hid_dim), (batch, seq_len, hid_dim), etc. we compute statistics across `hid_dim` then take the average
    of these statistics. For convenience we have included example metrics logged:

        +-------------------------------------------------------+-----------------------------------------------------+
        | Key                                                   | Logged data                                         |
        +=======================================================+=====================================================+
        |                                                       | The average max value of the `hid_dim` of the       |
        | ``activations/max/MODULE_NAME/input_{n}``             | nth input activations into the current module.      |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The average value of the `hid_dim` input            |
        | ``activations/average/MODULE_NAME/input_{n}``         | activations into the current module.                |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The average L2 Norm of the `hid_dim` of the         |
        | ``activations/l2_norm/MODULE_NAME/input_{n}``         | nth input activations into the current module.      |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The average kurtosis of the `hid_dim` of the nth    |
        | ``activations/kurtosis/MODULE_NAME/input_{n}``        | input activations into the current module.          |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The average max value of the `hid_dim` of the       |
        | ``activations/max/MODULE_NAME/output_{n}``            | nth ouput activations of the current module.        |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The average value of the `hid_dim` of the output    |
        | ``activations/average/MODULE_NAME/output_{n}``        | activations of the current module.                  |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The average L2 Norm of the values of the `hid_dim`  |
        | ``activations/l2_norm/MODULE_NAME/input_{n}``         | activations of the current module.                  |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The average kurtosis of the `hid_dim` of the nth    |
        | ``activations/kurtosis/MODULE_NAME/input_{n}``        | output activations of the current module.           |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+

    Args:
        interval (Union[int, str, Time], optional): Time string specifying how often to attach the logger and log the activations.
            For example, ``interval='5ba'`` means every 5 batches we log the activations. Default: '25ba'.
        ignore_module_types (Optional[List[str]], optional): A list of strings representing the class attributes we should ignore.
            For example passing in the list ['dropout', 'ln'] will cause the class attributes that contain
            'dropout' or 'ln' to not be logged. Default: 'None'.
        only_log_wandb (bool, optional): A bool that determines if we should only log to Weights and Biases. This is recommended
            in partcular for larger models as this callback logs a lot. Default: 'True'.
    """

    def __init__(self,
                 interval: Union[int, str, Time] = '25ba',
                 ignore_module_types: Optional[List[str]] = None,
                 keep_only_module_types: Optional[List[str]] = None,
                 only_log_wandb: bool = True,
                 log_module_output: bool = False,
                 log_gradients: bool = False,
                 log_grad_input: bool = False,
                 stat_list: Optional[List[str]] = ['amax', 'l2_norm', 'average']):
        self.ignore_module_types = ignore_module_types
        self.keep_only_module_types = keep_only_module_types
        self.only_log_wandb = only_log_wandb
        self.log_module_output = log_module_output
        self.log_gradients = log_gradients
        self.log_grad_input = log_grad_input
        self.stat_list = stat_list
        
        self.handles = []

        # Check that the interval timestring is parsable and convert into time object
        self.interval = Time.from_input(interval, TimeUnit.BATCH)

        if self.interval.unit == TimeUnit.BATCH and self.interval < Time.from_timestring('10ba'):
            warnings.warn(f'Currently the ActivationMonitor`s interval is set to {self.interval} '
                          f'which is below our recommended value of 10ba. We recommend you raise '
                          f'the interval to at least 10ba, as the activation monitor adds extra overhead '
                          f'and decreases throughput.')

        # Verify that the interval has supported units
        if self.interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f'Invalid time unit for parameter interval: '
                             f'{self.interval.unit}')

        self.last_train_time_value_logged = -1
        self.module_names = {}
        self.metrics = {}

        self.is_forward = False
        self.all_gather_buffer = None
        self.all_gather_valid_buffer = None

    def fit_start(self, state: State, logger: Logger):
        if not self.module_names:
            self.create_module_names(state.model)

        self.attach_hooks(state)
        log.info(
            "Attach logging hooks on modules: %s, %s",
            list(self.module_names.values()),
            self.handles,
        )
    
    @callback_timer(timer_type=TimerType.CPU, timer_level=1)
    def before_train_batch(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value

        if current_time_value % self.interval.value == 0 and \
            current_time_value != self.last_train_time_value_logged:

            global_actlog_state.set_enable_monitor_variable(True)
            self.is_forward = True

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def after_forward(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value

        if current_time_value % self.interval.value == 0 and \
            current_time_value != self.last_train_time_value_logged:
            
            self.is_forward = False

    @callback_timer(timer_type=TimerType.CPU, timer_level=2)
    def after_backward(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value

        if current_time_value % self.interval.value == 0 and \
            current_time_value != self.last_train_time_value_logged:
            self.last_train_time_value_logged = current_time_value

            self.make_log(logger, state.timestamp.batch.value)
            global_actlog_state.set_enable_monitor_variable(False)

            gc.collect()
            torch.cuda.empty_cache()

    def fit_end(self, state: State, logger: Logger):
        self.remove_hooks()
    
    def attach_hooks(self, state: State):
        self.register_hook(state.model)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        # Resetting handles we track
        self.handles = []

    def register_hook(self, model: torch.nn.Module):
        model.apply(self._register_hook)

    def _register_hook(self, module: torch.nn.Module):
        if not module in self.module_names: # skip filtered modules
            return
        self.handles.append(module.register_forward_hook(partial(self.base_hook, is_forward_hook=True)))
        if self.log_gradients:
            # Left it here, maybe we return hook implementation on newest torch?
            # self.handles.append(module.register_full_backward_hook(partial(self.base_hook, is_forward_hook=False)))

            module_name = self.module_names[module]
            module_name_normalized = replace_layer_pattern(module_name)
            layer_depends = module_name_normalized != module_name
            old_forward = module.forward
            def forward_patched(*input_tensor, **kwargs):
                if not self.is_forward or not global_actlog_state.get_enable_monitor_variable():
                    return old_forward(*input_tensor, **kwargs)
                
                if self.log_grad_input:
                    input_tensor = log_backward(
                        input_tensor, # input_tensor here is tuple
                        module_name=module_name_normalized,
                        grad_type="_grad_input",
                        layer_depends=layer_depends
                    )
                    input_tensor = ensure_tuple(input_tensor)
                    
                # input_tensor here is always tuple
                output = old_forward(*input_tensor, **kwargs)
                module_output_type = type(output)

                output = log_backward(
                    output, # output here is tuple or Tensor
                    module_name=module_name_normalized,
                    grad_type="_grad_output",
                    layer_depends=layer_depends
                )

                # output here is tuple or Tensor (based on module's output)
                assert module_output_type == type(output), \
                    f"Log hook brokes type of module output! {module_output_type} != {type(output)}"
                
                return output
            
            module.forward = forward_patched


    def filter_modules(self, module_name):
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

    def base_hook(self,
                  module: torch.nn.Module, 
                  input: Sequence,
                  output: Sequence,
                  is_forward_hook: bool):
        if not global_actlog_state.get_enable_monitor_variable():
            return 
        
        # Note: With activation checkpointing we need to check is it forward or
        # recomputation on backward
        if is_forward_hook and not self.is_forward:
            return
        
        module_name = self.module_names[module]
        metrics = self.metrics

        if self.is_forward:
            input_suffix = '_input'
            output_suffix = '_output'
            log_input, log_output = True, self.log_module_output
        else:
            input_suffix = '_grad_input'
            output_suffix = '_grad_output'
            log_input, log_output = self.log_grad_input, True

        if input is not None and log_input:
            for i, val in enumerate(input):
                if val is None or isinstance(val, dict):
                    continue
                if isinstance(val, str) and isinstance(input, dict):
                    self.recursively_add_metrics(metrics, module_name, f'{input_suffix}.{i}', input[val])  # type: ignore
                else:
                    self.recursively_add_metrics(metrics, module_name, f'{input_suffix}.{i}', val)

        if output is not None and log_output:
            for i, val in enumerate(output):
                if val is None or isinstance(val, dict):
                    continue
                if isinstance(val, str) and isinstance(output, dict):
                    self.recursively_add_metrics(metrics, module_name, f'{output_suffix}.{i}', output[val])  # type: ignore
                else:
                    self.recursively_add_metrics(metrics, module_name, f'{output_suffix}.{i}', val)

    def make_log(self, logger, step):
        # A rank that ran no expert module reports a different key set, so the keys
        # are unioned and each rank flags which of them it actually has.
        world_size = dist.get_world_size() or 1
        local_metrics = {}
        local_device = None
        for k, v in self.metrics.items():
            if not isinstance(v, torch.Tensor):
                continue
            if local_device is None:
                local_device = v.device
            if v.numel() > 0:
                local_metrics[k] = v.reshape(-1)[:1].to(dtype=torch.float32)

        if world_size > 1:
            gathered_keys = [list() for _ in range(world_size)]
            torch.distributed.all_gather_object(
                gathered_keys,
                sorted(local_metrics.keys()),
            )
        else:
            gathered_keys = [sorted(local_metrics.keys())]

        all_metric_keys = sorted({k for keys in gathered_keys for k in keys})
        metrics_count = len(all_metric_keys)
        if metrics_count == 0:
            self.metrics.clear()
            return

        if local_device is None:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_backend() == "nccl"
                and torch.cuda.is_available()
            ):
                local_device = torch.device("cuda", torch.cuda.current_device())
            else:
                local_device = torch.device("cpu")

        local_metric_vector = torch.stack(
            [
                local_metrics.get(
                    key,
                    torch.zeros((1,), dtype=torch.float32, device=local_device),
                )[0]
                for key in all_metric_keys
            ],
            dim=0,
        )
        local_valid_vector = torch.tensor(
            [1.0 if key in local_metrics else 0.0 for key in all_metric_keys],
            dtype=torch.float32,
            device=local_device,
        )

        if world_size > 1:
            world_size_x_metrics_count = world_size * metrics_count
            if (
                self.all_gather_buffer is None
                or self.all_gather_buffer.shape[0] != world_size_x_metrics_count
                or self.all_gather_buffer.device != local_device
            ):
                self.all_gather_buffer = torch.zeros(
                    world_size_x_metrics_count,
                    dtype=torch.float32,
                    device=local_device,
                )
            if (
                self.all_gather_valid_buffer is None
                or self.all_gather_valid_buffer.shape[0] != world_size_x_metrics_count
                or self.all_gather_valid_buffer.device != local_device
            ):
                self.all_gather_valid_buffer = torch.zeros(
                    world_size_x_metrics_count,
                    dtype=torch.float32,
                    device=local_device,
                )

            handle_values = torch.distributed.all_gather_into_tensor(
                self.all_gather_buffer,
                local_metric_vector,
                async_op=True,
            )
            handle_valid = torch.distributed.all_gather_into_tensor(
                self.all_gather_valid_buffer,
                local_valid_vector,
                async_op=True,
            )
            handle_values.wait()
            handle_valid.wait()
            gathered_values = self.all_gather_buffer.view(world_size, metrics_count)
            gathered_valid = self.all_gather_valid_buffer.view(world_size, metrics_count) > 0.5
        else:
            gathered_values = local_metric_vector.view(1, metrics_count)
            gathered_valid = local_valid_vector.bool().view(1, metrics_count)

        metrics = {}
        for i, k in enumerate(all_metric_keys):
            stat_values = gathered_values[:, i][gathered_valid[:, i]]
            if stat_values.numel() == 0:
                continue
            metrics[k] = compute_activation_stats(
                stat_values,
                k.split("/", 1)[0].rsplit("_", 1)[1],
            )
        if len(metrics) == 0:
            self.metrics.clear()
            return

        # metric aggregation by layer
        metric_aggr = defaultdict(list)
        for metric_name in metrics:
            for aggr_metric in ['max', 'min']:
                if aggr_metric in metric_name:
                    stat_name, module_name = metric_name.split("/", 1)
                    act_type, stat_type = stat_name.rsplit("_", 1)
                    module_name = replace_layer_pattern(module_name)
                    # act_type = activations | activations_gradients
                    # stat_type = amax | amin | max | min
                    # module_name = model.model.layers.{}.self_attn.o_proj_input.0 and etc.
                    aggr_key = (act_type, stat_type, module_name)
                    metric_aggr[aggr_key].append(metrics[metric_name])
        
        for (act_type, stat_type, module_name), values in metric_aggr.items():
            if len(values) == 1:
                continue
            stat_input = torch.cat(tuple(values))
            metrics[f"{act_type}/{stat_type}/{module_name}"] = compute_activation_stats(stat_input, stat_type)

        metrics = {k : v.item() for k, v in metrics.items()}
        
        if self.only_log_wandb:
            wandb_loggers = [ld for ld in logger.destinations if isinstance(ld, WandBLogger)]
            if len(wandb_loggers):
                for wandb_logger in wandb_loggers:
                    wandb_logger.log_metrics(metrics, step)
            else:
                # In the case there were no WandB loggers, just default to
                # the standard logger and let it take care of it
                logger.log_metrics(metrics)
        else:
            logger.log_metrics(metrics)

        self.metrics.clear()

    def recursively_add_metrics(self, metrics: dict, name: str, suffix: str, values: Any):
        # Becuase of the recursive diving, we need this call to prevent infinite recursion.
        if isinstance(values, str):
            return
        # Keep recursively diving if the value is a sequence
        if isinstance(values, Sequence):
            for i, value in enumerate(values):
                self.recursively_add_metrics(metrics, f'{name}_{i}', suffix, value)
            return
        else:
            self.add_metrics(metrics, name, suffix, values)

    def add_metrics(self, metrics: dict, name: str, suffix: str, value: torch.Tensor):
        # (veveselov): if value is Float8BlockwiseQTensor, we should skip this
        if type(value).__name__ == "Float8BlockwiseQTensor":
            return
            
        # We shouldn't log booleans
        if value.dtype == torch.bool:
            return
        if value.numel() == 0:
            return
        
        prefix = "activations" if self.is_forward else "activations_gradients"

        if value.is_floating_point() or value.is_complex():
            with torch.no_grad():
                for stat_name in self.stat_list:
                    activation_stat = compute_activation_stats(value, stat_name)
                    metric_name = f'{prefix}_{stat_name}/{name}{suffix}'
                    if metric_name in metrics:
                        metrics[metric_name].copy_(activation_stat)
                    else:
                        metrics[metric_name] = activation_stat

    def create_module_names(self, model: torch.nn.Module):
        self.module_names = {m: cleanup_metric_name(name) for name, m in model.named_modules() if self.filter_modules(name)}

def cleanup_metric_name(metric_name):
    metric_name = metric_name.replace("._module", "")
    metric_name = metric_name.replace(".root_module", "")
    metric_name = metric_name.replace("._orig_mod", "")
    metric_name = metric_name.replace("._checkpoint_wrapped_module", "")
    metric_name = metric_name.replace("._tp_linear_submodule", "")
    return metric_name


def compute_activation_stats(value: torch.Tensor, stat_name: str):
    if value.numel() == 0:
        return torch.full((1,), float("nan"), device=value.device, dtype=torch.float32)

    if stat_name == "amax": #absolute maximum
        output = value.abs().amax()
    elif stat_name == "amin": #absolute minimum
        output = value.abs().amin()
    elif stat_name == "max": #maximum
        output = value.amax()
    elif stat_name == "min": #minimum
        output = value.amin()
    elif stat_name == "amean": #absolute mean
        output = value.abs().mean()
    elif stat_name == "l2norm": # l2 norm
        output = torch.linalg.vector_norm(value, dim=-1).mean()
    elif stat_name == "average": # average
        output = value.mean()
    elif stat_name == "kurtosis": # kurtosis
        output = compute_kurtosis(value)
    elif stat_name.startswith("actrel"):
        op = stat_name[len("actrel")]
        threshold = float(stat_name[len("actrel") + 1:])
        abs_val = value.abs() 
        mean_abs = abs_val.mean()  
        if op == ">":
            output = abs_val.gt_(threshold * mean_abs).mean()
        elif op == "<":
            output = abs_val.lt_(threshold * mean_abs).mean()
        else:
            raise ValueError(f"This statistic {stat_name} is not supported!")
    elif stat_name.startswith("actabs"):
        op = stat_name[len("actabs")]  
        threshold = float(stat_name[len("actabs") + 1:])
        if op == ">":
            output = value.abs().gt_(threshold).mean()
        elif op == "<":
            output = value.abs().lt_(threshold).mean()
        else:
            raise ValueError(f"This statistic {stat_name} is not supported!")
    else:
        raise ValueError(f"This statistic {stat_name} is not supported!")

    return output.view(1)

def compute_kurtosis(value: torch.Tensor):
    # Computes the kurtosis over the last dimension
    mean = torch.mean(value, dim=-1).unsqueeze(-1)
    diffs = value - mean
    m_4 = torch.mean(torch.pow(diffs, 4), dim=-1)
    var = torch.mean(torch.pow(diffs, 2), dim=-1)
    return (m_4 / (var**2)).mean()
