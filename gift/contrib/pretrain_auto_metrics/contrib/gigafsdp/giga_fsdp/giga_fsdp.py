import gc
import os
import socket
import time
import warnings
import re
import contextlib
from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Type, Optional, Union, Dict, Tuple, List, TypeVar
from packaging import version
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.distributed.fsdp.wrap
import torch.nn as nn
from torch.distributed.fsdp._init_utils import (
    _materialize_with_param_init_fn, # non-patched version
    _move_module_to_device,
    _need_to_materialize_module,
)
from torch.distributed.utils import _sync_module_states
from torch.distributed.fsdp._init_utils import PARAM_BROADCAST_BUCKET_SIZE
from torch.utils._pytree import tree_map_only

from .meta_param import MetaParam, convert_some_params_to_metaparams, materialize_params
from .float8_utils import (
    FP8_TILE_SIZE,
    Fp8ParamGeom,
    compute_local_tile_amaxes,
    fold_and_quantize_param,
    fp8_param_shape_is_supported,
    is_expert_param,
)
from .common import (
    ParamBufferPrefix,
    MetaInfoKeys,
    EMA_SUFFIX,
    CHECKPOINT_VERSION
)
from .optimizer_utils import fill_optimizer_meta_info
from .logger import logger
from .pg_manager import ProcessGroupManager, ProcessGroupKit

T = TypeVar("T", bound="Module")

GRAD_NAN_SLOTS = 1024

# Armed by the optimizer's nan-skip branch (see report_grad_nan_origin_and_arm), or from the
# start with GIGAFSDP_GRAD_NAN_ORIGIN=1. Disarmed by default, so a healthy run runs no extra code.
_grad_nan_armed = os.environ.get("GIGAFSDP_GRAD_NAN_ORIGIN", "0") == "1"
_grad_nan_flags = None          # int32 device counter, one slot per super-tensor
_grad_nan_names: List[str] = []  # slot -> a param fqn of that super-tensor
_grad_nan_pending = False        # flags hold data that has not been read out yet
_grad_nan_reports = 0


def _grad_nan_where() -> str:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    per_node = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("NGPUS_PER_NODE")
                   or torch.cuda.device_count() or 1)
    return (f"host={socket.gethostname()} node={rank // per_node if rank >= 0 else '?'} "
            f"node_rank={os.environ.get('NODE_RANK', '?')} global_rank={rank} "
            f"local_rank={os.environ.get('LOCAL_RANK', '?')} "
            f"cuda={torch.cuda.current_device() if torch.cuda.is_available() else -1}")


def accumulate_local_grad_nan(super_tensor, grad: torch.Tensor) -> None:
    """Flag a non-finite LOCAL gradient, without a host sync.

    Called on the comm stream right before reduce_scatter -- the last point where the
    gradient is still purely this rank's own. After the collective every rank holds a
    shard summed over the whole group, so the optimizer's "Skip step because grads had
    nan" can only count devices; worse, the shard lands on its owner, not on the rank
    that produced the nan, so the count points at an innocent rank.

    Writes into a device counter only: reading it here would make the CPU drain the comm
    stream at every super-tensor and destroy the backward/communication overlap.
    """
    global _grad_nan_flags, _grad_nan_pending
    if _grad_nan_flags is None:
        _grad_nan_flags = torch.zeros(GRAD_NAN_SLOTS, dtype=torch.int32, device=grad.device)
    slot = getattr(super_tensor, "_grad_nan_slot", None)
    if slot is None:
        if len(_grad_nan_names) >= GRAD_NAN_SLOTS:
            return
        slot = len(_grad_nan_names)
        _grad_nan_names.append(str(next(iter(super_tensor.param_fqn_to_offset), "?")))
        super_tensor._grad_nan_slot = slot
    _grad_nan_flags[slot] += (~torch.isfinite(grad)).any()
    _grad_nan_pending = True


def report_grad_nan_origin_and_arm(max_reports: int = 20) -> None:
    """Read the counters out and arm the next step, naming EVERY offending rank.

    Called from the optimizer branch that skips a step because gradients were non-finite.
    That branch is entered by every rank (its condition comes from an all_reduce), so one
    small all_reduce here is safe and lets each rank print the complete list of sources
    instead of only its own verdict -- with several bad nodes you get all of them in one
    line, in every log, rather than one line per rank scattered over per-node files.

    The first skip can only arm: the step that produced the nan is already reduced and
    gone. Non-finite local grads always reach some shard and therefore always cause a
    skip, so the counters are read out in the same step they were filled.

    Collectives are gated on `_grad_nan_armed` only -- it is changed nowhere else, so it
    stays identical on every rank. Printing is gated on local state; printing cannot
    hang, a collective can. Stops after `max_reports` calls so a nan storm cannot flood
    the log or keep paying for the scan (~2.5 ms per GB of gradient buffer).
    """
    global _grad_nan_armed, _grad_nan_pending, _grad_nan_reports
    if not _grad_nan_armed:
        _grad_nan_armed = _grad_nan_reports < max_reports
        return

    bad = []
    if _grad_nan_flags is not None:
        flags = _grad_nan_flags[: len(_grad_nan_names)].cpu()   # the one sync, once per step
        bad = [_grad_nan_names[i] for i in range(flags.numel()) if flags[i]]
        _grad_nan_flags.zero_()

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    sources = []
    if distributed:
        world = dist.get_world_size()
        device = _grad_nan_flags.device if _grad_nan_flags is not None else None
        counts = torch.zeros(world, dtype=torch.int32, device=device)
        counts[rank] = len(bad)
        dist.all_reduce(counts)
        sources = [(r, int(counts[r])) for r in range(world) if int(counts[r])]

    if _grad_nan_pending or bad:
        per_node = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("NGPUS_PER_NODE")
                       or torch.cuda.device_count() or 1)
        if rank == 0:
            if sources:
                nodes = sorted({r // per_node for r, _ in sources})
                detail = ", ".join(f"rank{r}(node{r // per_node}, {n} super-tensors)"
                                   for r, n in sources[:16])
                print(f"[grad-nan] SOURCES: {len(sources)} rank(s) on node(s) {nodes} had "
                      f"non-finite gradients BEFORE reduce_scatter: {detail}"
                      f"{' ...' if len(sources) > 16 else ''}", flush=True)
            else:
                print("[grad-nan] no rank reported a non-finite gradient before "
                      "reduce_scatter -- the nan did not come from the gradient reduction",
                      flush=True)
        if bad:
            print(f"[grad-nan] this rank is a SOURCE: {len(bad)} super-tensor(s), first: "
                  f"{bad[:4]}{' ...' if len(bad) > 4 else ''} {_grad_nan_where()}", flush=True)
        _grad_nan_pending = False

    _grad_nan_reports += 1
    _grad_nan_armed = _grad_nan_reports < max_reports


def get_torch_version() -> version.Version:
    torch_version = re.match(r"^(\d+\.\d+\.\d+(\.[a-z]+)?)", torch.__version__).group()
    return version.parse(torch_version)


if get_torch_version() < version.parse("2.4"):
    _move_module_to_device_impl = _move_module_to_device
    def _move_module_to_device(module, ignored_params, ignored_buffers, device_from_device_id):
        assert not ignored_buffers
        return _move_module_to_device_impl(module=module, ignored_params=ignored_params, device_from_device_id=device_from_device_id)

    _materialize_with_param_init_fn_impl = _materialize_with_param_init_fn
    def _materialize_with_param_init_fn(root_module, param_init_fn, ignored_modules):
        assert not ignored_modules
        return _materialize_with_param_init_fn_impl(root_module=root_module, param_init_fn=param_init_fn)


def _init_module(module_ : nn.Module,
                 param_init_fn: Optional[Callable] = None,
                 sync_module_states: bool = False,
                 fsdp_process_group: Optional[dist.ProcessGroup]  = None,
                 device: torch.device = None
                 ):

    is_meta_module, _ = _need_to_materialize_module(module_, set(), set())
    if is_meta_module:
        _materialize_with_param_init_fn(
            root_module=module_,
            param_init_fn=param_init_fn,
            ignored_modules=set(),
        )
    _move_module_to_device(
        module=module_,
        ignored_params=set(),
        ignored_buffers=set(),
        device_from_device_id=device,
    )

    if sync_module_states:
        _sync_module_states(
            module=module_,
            process_group=fsdp_process_group,
            broadcast_bucket_size=PARAM_BROADCAST_BUCKET_SIZE,
            src=0,
            params_and_buffers_to_ignore=set(),
        )


class PrefetchHook:
    def __init__(self, prefetch_enable: bool = True):
        self._first_module: Optional[nn.Module] = None
        self._last_called_module: Optional[nn.Module] = None
        self._order_mapping: Dict["SuperTensorModule", "SuperTensorModule"] = dict()
        self.prefetch_enable = prefetch_enable

    def __call__(self, module: "SuperTensorModule", *args: Tuple, **kwargs: Dict[Any, Any]) -> None:
        if module.super_tensor.first_time_processed is None:
            module.super_tensor.first_time_processed = time.time()

        if self._first_module is None:
            self._first_module = module

        if module is not self._first_module:
            self._order_mapping[self._last_called_module] = module

        self._last_called_module = module

        next_module = self._order_mapping.get(module, None)
        # if sequential modules have the same reusable_buffer, we can't do prefetch
        # because this requires making an all-gather of 2 layers in the same buffer at the same time
        is_next_buffer_same = self.check_buffer_order(module, next_module)
        if next_module is not None and self.prefetch_enable and not is_next_buffer_same:
            next_super_tensor = next_module.super_tensor

            # previous layer have already called end_processing and we can use this buffer
            if next_super_tensor.weight_reusable_buffer.reusable_buffer._released_event is not None:
                logger.debug(f"Call prefetch from {module.name} for {next_module.name}")
                next_super_tensor.all_gather(next_super_tensor.comm_stream)

    def check_buffer_order(self,
                           module: "SuperTensorModule",
                           next_module: Optional["SuperTensorModule"]) -> None:
        if next_module is None:
            return
        current_buffer = module.super_tensor.weight_reusable_buffer.reusable_buffer
        next_buffer = next_module.super_tensor.weight_reusable_buffer.reusable_buffer

        return current_buffer is next_buffer


class ExecutionOrderTracker:
    def __init__(self, prefetch_enable: bool = True):
        self._forward_pre_hook = PrefetchHook(prefetch_enable)
        self._backward_pre_hook = PrefetchHook(prefetch_enable)
        self._module_to_force_input = dict()

    def track(self, module: nn.Module) -> None:
        module.register_forward_pre_hook(self._drop_stale_forced_inputs)
        module.register_forward_pre_hook(self._forward_pre_hook)
        module.register_full_backward_pre_hook(self._backward_pre_hook)

    def _drop_stale_forced_inputs(self, module: nn.Module, args) -> None:
        """Forced inputs must not outlive the forward pass that filed them.

        A module invoked more than once per forward -- a recurrent head applied
        at every depth -- files a forced input on its last call whose consumer
        never runs again this step, because `order_mapping` maps it to itself.
        Left in place it is popped on the NEXT step, and `TyingGate` then sews a
        fresh graph onto the previous iteration's freed one ("backward through
        the graph a second time"). Nothing legitimate survives a forward: the
        last module of a pass has no recorded successor, so it files nothing,
        which makes this a no-op wherever each module is called once.
        """
        if module is self._forward_pre_hook._first_module:
            self._module_to_force_input.clear()

    def get_next_fwd_module(
            self,
            super_tensor_module : "SuperTensorModule"
        ) -> "SuperTensorModule":
        return self._forward_pre_hook._order_mapping.get(super_tensor_module, None)

    def add_force_module_input(
            self,
            super_tensor_module : "SuperTensorModule",
            input: Union[torch.Tensor, tuple[torch.Tensor]]
        ) -> None:
        next_module = self.get_next_fwd_module(super_tensor_module)

        if next_module is None:
            return

        if isinstance(input, torch.Tensor):
            input = (input,)

        # we don't need to force input without requires_grad
        # because in this case autograd will not compute backward throw this module
        forced_input = tuple(
            x for x in input
            if isinstance(x, torch.Tensor) and x.requires_grad == True
        )
        if len(forced_input):
            self._module_to_force_input[next_module] = forced_input

    def get_force_module_input(
            self,
            super_tensor_module : "SuperTensorModule"
        ) -> tuple[Optional[torch.Tensor]]:
        return self._module_to_force_input.pop(super_tensor_module, tuple())


class ReusableBufferViewState(Enum):
    NOT_READY = 1
    READY = 2
    IN_PROCESS = 3


def is_main_parameter_trainable(module):
    return any([module_.super_tensor.is_trainable
                for module_ in module.modules()
                if isinstance(module_, SuperTensorModule)])


def module_params_requires_grad(module : nn.Module) -> List[bool]:
    requires_grad_list = [param.requires_grad for param in module.parameters()]
    return requires_grad_list


def divide_param_grad(
    param : nn.Parameter,
    pg_kit : ProcessGroupKit,
    gradient_accumulation_steps : int
    ):
    divider = pg_kit.fsdp_process_group.size() * gradient_accumulation_steps
    if pg_kit.inter_hybrid_process_group:
        divider *= pg_kit.inter_hybrid_process_group.size()

    assert param.grad is not None, "Gradient is not computed for this parameter"
    param.grad.data.div_(divider)


class GigaFSDP(nn.Module):
    """A wrapper for sharding module parameters with GigaFSDP.

    Args:
        module (nn.Module):
            Module to be wrapped.
        zero_stage (int):
            Type of sharding to apply (analogous to `sharding_strategy` in
            FSDP). `3` is for ZeRO-3 like sharding, 2 is for `ZeRO-2 like
            sharding.
        param_dtype (torch.dtype):
            Dtype to use during forward pass.
        modules_to_wrap_with_names (list[tuple[nn.Module, str]]):
            Tuple of modules to shard with corresponding names for state_dict
            (analogous to `auto_wrap_policy` in FSDP).
        layer_norm_module_cls (Type[nn.Module]):
            Class of layer norm layers.
        gradient_accumulation_steps (int):
            Number of gradient accumulation steps.
        fsdp_process_group (dist.ProcessGroup | None, optional):
            Process group to shard parameters across.
            Defaults to None.
        intra_node_fsdp_process_group (dist.ProcessGroup | None, optional):
            A group of processes which share the node host and `data_parallel_process_group`.
            Defaults to None.
        model_parallel_process_group (dist.ProcessGroup | None, optional):
            A group of tensor parallel processes. Defaults to None.
        all_reduce_grads_across_model_parallel_group (bool, optional):
            If True, layer norm parameters are reduced across `model_parallel_process_group`.
            Defaults to False.
        bit16_reduce_scatter (bool, optional):
            If True, reduce scatter is performed in bfloat16.
            Defaults to False.
        bit32_acc_for_bit16_reduce_scatter (bool, optional):
            If True, uses a custom kernel for float32 accuracy for bfloat16 reduce scatter. Defaults to False.
        hpz_first_layers_num (int, optional):
            Number of layers to apply HPZ to.
            Defaults to 0.
        rogue_layer_norm_modules_with_names (dict[nn.Module, str] | None, optional):
            Mapping of modules which contain only layer norm parameters to
            their corresponding names for state_dict.
            Defaults to None.
        sync_module_states (bool, optional):
            If True, module states are synced across
            `intra_node_data_parallel_process_group` before sharding.
            Defaults to False.
        param_init_fn (Callable | None, optional):
            A function to initialize modules before sharding (and before syncing).
            Defaults to None.
        device_id (int | None, optional):
            Device to use for initialization and sharding.
            Defaults to None.
        hybrid_group_size (int | None, optional):
            Size of hybrid shard group. Must divide fsdp_process_group.size().
            Defaults to None.
        hpz_group_size (int | None, optional):
            Size of hzp sharding group. Must divide fsdp_process_group.size().
            Defaults to None.
    """
    def __init__(
        self,
        module: nn.Module,
        zero_stage: int,
        param_dtype: torch.dtype,
        modules_to_wrap_with_names: list[tuple[nn.Module, str]],
        gradient_accumulation_steps: int,
        layer_norm_module_cls: Optional[nn.Module] = None,
        fsdp_process_group: Optional[dist.ProcessGroup]  = None,
        model_parallel_process_group: Optional[dist.ProcessGroup] = None,
        special_process_group_fn: Optional[Callable] = None,
        all_reduce_grads_across_model_parallel_group: bool = False,
        bit16_reduce_scatter: bool = False,
        bit32_acc_for_bit16_reduce_scatter: bool = False,
        hpz_first_layers_num: int = 0,
        rogue_layer_norm_modules_with_names: Optional[dict[nn.Module, str]] = None,
        sync_module_states: bool = False,
        param_init_fn: Optional[Callable] = None,
        device_id: Optional[int] = None,
        hybrid_group_size: Optional[int] = None,
        hpz_group_size: Optional[int] = None,
        prefetch_enable: bool = True,
        buffer_count: int = 2,
        alignment_value: Optional[int] = None,
        float8_all_gather_on_experts: bool = False,
        float8_all_gather_skip_patterns: Optional[List[str]] = None,
    ):
        super().__init__()

        if zero_stage == 1: warnings.warn("It is better to use zero_stages 2 or 3.")
        assert not bit32_acc_for_bit16_reduce_scatter, (
            "we do not support rs in bfp16 with acc32 at the moment"
        )

        self._module = module
        self._zero_stage = zero_stage
        self._param_dtype = param_dtype
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._default_gradient_accumulation_steps = gradient_accumulation_steps
        self._layer_norm_module_cls = layer_norm_module_cls
        self._separate_layer_norm_enabled = layer_norm_module_cls is not None
        self._all_reduce_grads_across_model_parallel_group = all_reduce_grads_across_model_parallel_group
        self._bit32_acc_for_bit16_reduce_scatter = bit32_acc_for_bit16_reduce_scatter
        self._device = torch.device(torch.cuda.current_device() if device_id is None else device_id)
        self._backwards_count = 0
        self._order_inited = False
        self.has_ema_flag = False
        self._float8_all_gather_on_experts = float8_all_gather_on_experts

        self._float8_all_gather_skip_patterns: List[str] = list(
            float8_all_gather_skip_patterns or []
        )
        if float8_all_gather_on_experts:
            assert alignment_value == 128, (
                f"float8_all_gather_on_experts requires alignment_value=128, got {alignment_value=}"
            )
            assert zero_stage == 3, (
                "blockwise all-gater for fp8 experts support only zero stage 3!"
            )

        self._alignment_value = alignment_value
        assert self._alignment_value is None or self._alignment_value > 0, f"alignment value must be positive, but {self._alignment_value=}"
        # cuda streams
        self._compute_stream = torch.cuda.current_stream()
        self._comm_stream = torch.cuda.Stream(priority=-1)
        self._hpz_all_gather_stream = torch.cuda.Stream(priority=-1) if hpz_first_layers_num > 0 else None

        requires_grad_list = module_params_requires_grad(module)
        if any(requires_grad_list): #there are trainable params
            self.is_inference_mode = False
        else: # all params are frozen
            self.is_inference_mode = True
            logger.debug("GigaFSDP is_inference_mode=True. There is no freezed_parameters buffer!")

        self.pg_manager = ProcessGroupManager(
            fsdp_process_group,
            model_parallel_process_group,
            hybrid_group_size,
            hpz_group_size
        )

        exec_order_tracker = ExecutionOrderTracker(prefetch_enable)

        target_module_to_kwargs = dict()
        for idx, (m, name) in enumerate(modules_to_wrap_with_names):
            pg_kit = self._get_pg_kit_for_module(m, self.pg_manager, special_process_group_fn)
            
            num_params = sum(p.numel() for p in m.parameters())
            logger.info(f"Layer {type(m)=} {num_params=} {name=} wrapped with {pg_kit=}")

            is_hpz_layer = False
            if hpz_first_layers_num > idx and pg_kit.intra_node_fsdp_process_group is not None:
                is_hpz_layer = True
            elif hpz_first_layers_num > idx:
                # for tp_size >= device_count we don't create hpz subgroups and disable it
                tp_size = pg_kit.model_parallel_process_group.size() if pg_kit.model_parallel_process_group else 1
                device_count = torch.cuda.device_count()
                logger.info(f"tp_size={tp_size} >= device_count={device_count}, disable hpz for {name}")

            target_module_to_kwargs[m] = {
                "comm_stream": self._comm_stream,
                "pg_kit": pg_kit,
                "is_hpz_layer": is_hpz_layer,
                "hpz_all_gather_stream": self._hpz_all_gather_stream,
                "name": name,
                "sync_module_states": sync_module_states,
                "param_init_fn": param_init_fn,
                "layer_norm_module_cls": self._layer_norm_module_cls,
                "device": self._device,
                "exec_order_tracker": exec_order_tracker,
                "zero_stage": self._zero_stage,
                "alignment_value": self._alignment_value,
                "fp8_all_gather_selected": self._is_fp8_selected(name),
                "param_dtype": param_dtype,
            }

        torch.distributed.fsdp.wrap._post_order_apply(
            self._module,
            fn=torch.distributed.fsdp.wrap._construct_wrap_fn(
                self._module,
                target_module_to_kwargs=target_module_to_kwargs,
                fsdp_fn=SuperTensorModule,
            ),
        )

        # Meta stored in every process conversions. Stored in state dict as well.
        self._setup_meta_info()

        self._super_tensors: list[SuperTensor] = []
        self._all_buffers: list[ReusableBuffer] = []
        self._is_separate_module_trainable = None
        self._is_main_parameter_trainable = is_main_parameter_trainable(self._module)

        separate_module_metaparms : List[MetaParam] = []
        max_buffer_size = 0
        max_grad_buffer_size = 0
        sum_shards = 0
        sum_shards_freezed = 0

        for module_ in self._module.modules():
            if rogue_layer_norm_modules_with_names is not None and module_ in rogue_layer_norm_modules_with_names:
                _init_module(module_,
                             param_init_fn=param_init_fn,
                             sync_module_states=sync_module_states,
                             fsdp_process_group=self.pg_manager.default_pg_kit.fsdp_process_group,
                             device=self._device
                             )
                layer_separate_module_metaparms = convert_some_params_to_metaparams(
                    module_, rogue_layer_norm_modules_with_names[module_], lambda _: True
                )
                separate_module_metaparms.extend(layer_separate_module_metaparms)
            elif isinstance(module_, SuperTensorModule):
                self._super_tensors.append(module_.super_tensor)

                max_buffer_size = max(max_buffer_size, module_.super_tensor.weight_buffer_size)
                max_grad_buffer_size = max(max_grad_buffer_size, module_.super_tensor.grad_buffer_size)
                sum_shards += module_.super_tensor.shard_size
                sum_shards_freezed += module_.super_tensor.shard_size_freezed

                layer_separate_module_metaparms = convert_some_params_to_metaparams(
                    module_.root_module, module_.name, lambda _: True
                )
                separate_module_metaparms.extend(layer_separate_module_metaparms)

                self._add_info_to_metainfo(module_.super_tensor, module_.super_tensor.params, ParamBufferPrefix.TRAINABLE)
                self._add_info_to_metainfo(module_.super_tensor,
                                           module_.super_tensor.params_freezed,
                                           ParamBufferPrefix.FREEZED
                                           if self._is_main_parameter_trainable else ParamBufferPrefix.TRAINABLE)

        self._create_separate_module_super_tensor(separate_module_metaparms)
        # count of SuperTensors without optional layernorm_supertensor
        self._main_st_count : int = len(self._super_tensors)
        if self._separate_layer_norm_enabled:
            self._main_st_count -= 1

        ## Create parameters of GigaFSDP module

        # if sum_shards != 0 -> main_param_count := sum_shards
        # if sum_shards == 0 (all freezed) -> main_param_count := sum_shards_freezed
        # for ALL parameter buffers we use additional *freeze* buffers
        # ONLY if they are really needed (we can't keep params in one buffer with requires_grad=True|False).
        main_param_count = sum_shards or sum_shards_freezed
        self.register_buffer(f'_{ParamBufferPrefix.TRAINABLE}_parameter', torch.empty(
                main_param_count,
                dtype=torch.float if self._is_main_parameter_trainable else param_dtype,
                device=self._device,
                requires_grad=self._is_main_parameter_trainable
            ),
            persistent=False
        )

        self._main_parameter.grad = (
            torch.zeros(
                main_param_count,
                dtype=torch.float,
                device=self._device,
            )
            if self._is_main_parameter_trainable
            else None
        )
        self._main_parameter_bit16 = (
            torch.empty(
                main_param_count,
                dtype=param_dtype,
                device=self._device,
            )
            if self._is_main_parameter_trainable
            else self._main_parameter.data # *.data is tensor, not nn.Parameter
        )

        if self._is_main_parameter_trainable and sum_shards_freezed:
            self.register_buffer(f'_{ParamBufferPrefix.FREEZED}_parameter', torch.empty(
                    sum_shards_freezed,
                    dtype=param_dtype,
                    device=self._device,
                    requires_grad=False
                ),
                persistent=False
            )
            self._freezed_parameter.grad = None
            self._freezed_parameter_bit16 = self._freezed_parameter.data
        else:
            self._freezed_parameter = None
            self._freezed_parameter_bit16 = None

        if self._separate_layer_norm_enabled:
            sep_param_count = self._layernorm_supertensor.shard_size or self._layernorm_supertensor.shard_size_freezed
            self._layernorm_parameter = torch.nn.Parameter(
                torch.empty(
                    sep_param_count,
                    dtype=torch.float if self._is_separate_module_trainable else param_dtype,
                    device=self._device,
                ),
                requires_grad=self._is_separate_module_trainable
            )
            self._layernorm_parameter.grad = (
                torch.zeros(
                    sep_param_count,
                    dtype=torch.float,
                    device=self._device,
                )
                if self._is_separate_module_trainable
                else None
            )
            self._layernorm_parameter_bit16 = (
                torch.empty(
                    sep_param_count,
                    dtype=param_dtype,
                    device=self._device,
                )
                if self._is_separate_module_trainable
                else self._layernorm_parameter.data
            )
            if self._is_separate_module_trainable and self._layernorm_supertensor.shard_size_freezed:
                self._layernormfreezed_parameter = torch.nn.Parameter(
                    torch.empty(
                        self._layernorm_supertensor.shard_size_freezed,
                        dtype=param_dtype,
                        device=self._device,
                    ),
                    requires_grad=False
                )
                self._layernormfreezed_parameter.grad = None
                self._layernormfreezed_parameter_bit16 = self._layernormfreezed_parameter.data
            else:
                self._layernormfreezed_parameter_bit16 = None

        ## Setup ReusableBuffers
        if not bit16_reduce_scatter and (self._is_main_parameter_trainable or self._is_separate_module_trainable):
            layernorm_grad_buffer_size = self._layernorm_supertensor.grad_buffer_size if self._is_separate_module_trainable else 0
            self._fp32_grad_buffer = ReusableBuffer(
                max(layernorm_grad_buffer_size, max_grad_buffer_size),
                device="cuda",
                dtype=torch.float,
                prepare_stream=self._comm_stream,
                process_stream=self._comm_stream,
            )
            self._all_buffers.append(self._fp32_grad_buffer)
        else:
            self._fp32_grad_buffer = None

        self._weight_buffers = [
            ReusableBuffer(
                max_buffer_size if zero_stage == 3 else self._super_tensors[i].weight_buffer_size,
                device="cuda",
                dtype=param_dtype,
                prepare_stream=self._comm_stream,
                process_stream=self._compute_stream,
            )
            for i in range(buffer_count if zero_stage == 3 else self._main_st_count)
        ]

        self._grad_buffers = [
            ReusableBuffer(
                max(max_buffer_size, max_grad_buffer_size)
                if zero_stage > 1 else self._super_tensors[i].grad_buffer_size,
                device="cuda",
                dtype=param_dtype,
                prepare_stream=self._compute_stream,
                process_stream=self._comm_stream,
                fill_zeros=True,
            )
            if (self._is_main_parameter_trainable if zero_stage > 1 else self._super_tensors[i].is_trainable)
            else None
            for i in range(buffer_count if zero_stage > 1 else self._main_st_count)
        ]

        self._all_buffers.extend(self._weight_buffers)
        self._all_buffers.extend(self._grad_buffers if self._is_main_parameter_trainable else [])
        self._all_buffers = [buffer for buffer in self._all_buffers if buffer is not None]

        ## Assign buffers to SuperTensors
        shard_buffer_offset = 0
        shard_buffer_offset_freezed = 0
        for idx, module_ in enumerate(m for m in self._module.modules() if isinstance(m, SuperTensorModule)):
            module_.super_tensor.set_buffers(
                self._weight_buffers[idx % len(self._weight_buffers)]
                if self._zero_stage < 3 else self._weight_buffers[0],
                self._grad_buffers[idx % len(self._grad_buffers)]
                if self._zero_stage < 2 else self._grad_buffers[0],
            )
            module_.super_tensor.set_fp32_grad_buffer(self._fp32_grad_buffer)

            # Note: if inference mode is enabled, the weights are stored in _main_parameter buffer
            module_.super_tensor.set_weight_shard_buffers(
                    self._main_parameter_bit16,
                    self._freezed_parameter_bit16 if self._is_main_parameter_trainable else self._main_parameter_bit16,
                    shard_buffer_offset,
                    shard_buffer_offset_freezed,
                    self._main_parameter
                )

            module_.init_layerwise_params(self._main_parameter, self._freezed_parameter)

            shard_buffer_offset += module_.super_tensor.shard_size
            shard_buffer_offset_freezed += module_.super_tensor.shard_size_freezed

            if bit32_acc_for_bit16_reduce_scatter:
                module_.super_tensor.enable_bit32_acc_for_bit16_reduce_scatter()

        if self._float8_all_gather_on_experts:
            self._setup_fp8_expert_state()

        self._setup_tied_modules()
        self._init_tied_modules()

        self._set_buffers_to_separate_module_super_tensor()

        ## Register Hooks
        self.register_forward_pre_hook(self._get_forward_pre_hook())
        if self._separate_layer_norm_enabled:
            self.register_forward_hook(self._layernorm_supertensor.get_forward_hook())
            self.register_full_backward_pre_hook(self._layernorm_supertensor.get_backward_pre_hook())
        self.register_load_state_dict_pre_hook(self._get_load_state_dict_pre_hook())
        self.register_load_state_dict_post_hook(self._get_load_state_dict_post_hook())

        self._add_global_metainfo(special_process_group_fn=special_process_group_fn)

        logger.info(self.get_memory_stats())

        gc.collect()
        torch.cuda.empty_cache()

    def setup_ema_buffer(self):
        if self._is_main_parameter_trainable:
            self.register_buffer(
                f'_main_parameter{EMA_SUFFIX}',
                torch.empty_like(self._main_parameter.data),
                persistent=False
            )
        if self._is_separate_module_trainable:
            self.register_buffer(
                f'_layernorm_parameter{EMA_SUFFIX}',
                torch.empty_like(self._layernorm_parameter.data),
                persistent=True
            )

        for module_ in self._module.modules():
            if isinstance(module_, SuperTensorModule):
                module_.init_layerwise_params(self._main_parameter_ema, None, skip_freezed_params=True, suffix=EMA_SUFFIX)

    def update_ema_buffer_after_load(self, force_update = False):
        if force_update or not self.has_ema_flag:
            if self._is_main_parameter_trainable:
                self._main_parameter_ema.data.copy_(self._main_parameter.data)
            if self._is_separate_module_trainable:
                self._layernorm_parameter_ema.data.copy_(self._layernorm_parameter.data)

    def update_ema_params(self, alpha: float):
        if self._is_main_parameter_trainable:
            self._main_parameter_ema.data.lerp_(self._main_parameter.data, alpha)
        if self._is_separate_module_trainable:
            self._layernorm_parameter_ema.data.lerp_(self._layernorm_parameter.data, alpha)

    def swap_ema_and_weights(self):
        if self._is_main_parameter_trainable:
            self._main_parameter.data, self._main_parameter_ema.data = self._main_parameter_ema.data, self._main_parameter.data
        if self._is_separate_module_trainable:
            self._layernorm_parameter.data, self._layernorm_parameter_ema.data = self._layernorm_parameter_ema.data, self._layernorm_parameter.data

    def _is_fp8_selected(self, module_name: str) -> bool:
        if not self._float8_all_gather_on_experts:
            return False
        if any(pat in module_name for pat in self._float8_all_gather_skip_patterns):
            logger.info(f"[fp8] {module_name}: skipped by float8_all_gather_skip_patterns")
            return False
        return True

    def _fp8_super_tensors(self) -> List["SuperTensor"]:
        return [st for st in self._super_tensors[:self._main_st_count] if st.is_fp8]

    def _setup_fp8_expert_state(self) -> None:
        fp8_sts = self._fp8_super_tensors()
        if not fp8_sts:
            return None

        expert_max_padded_numel = max(st.padded_numel for st in fp8_sts)
        self._expert_bf16_scratch = torch.full(
            (expert_max_padded_numel,),
            float("nan"),
            dtype=self._param_dtype,
            device=self._device,
        )

        groups = {st._fsdp_process_group for st in fp8_sts}
        assert len(groups) == 1, (
            f"[fp8] all fp8 SuperTensors must share one process group, found {len(groups)} "
            f"across {[st.name for st in fp8_sts]}"
        )
        self._fp8_group = groups.pop()

        fp8_sts = sorted(fp8_sts, key=lambda st: st.name)
        self._fp8_sts = fp8_sts

        packed_scale_stride = sum(st.shard_size // FP8_TILE_SIZE for st in fp8_sts)
        self._meta_info["fp8_packed_amax_layout"] = [
            [st.name, st.shard_size // FP8_TILE_SIZE] for st in fp8_sts
        ]
        self._meta_info["meta_for_hash"].append("fp8_packed_amax_layout")
        self._expert_local_amaxes = torch.empty(
            packed_scale_stride, dtype=torch.float32, device=self._device
        )
        self._expert_gathered_amaxes = torch.empty(
            packed_scale_stride * self._fp8_group.size(),
            dtype=torch.float32, device=self._device,
        )

        scale_offset = 0
        for st in fp8_sts:
            n = st.shard_size // FP8_TILE_SIZE
            st.init_fp8_expert_state(
                local_amaxes=self._expert_local_amaxes[scale_offset : scale_offset + n],
                gathered_amaxes=self._expert_gathered_amaxes,
                scale_offset=scale_offset,
                packed_scale_stride=packed_scale_stride,
                bf16_scratch=self._expert_bf16_scratch,
                device=self._device,
            )
            scale_offset += n

    def _setup_tied_modules(self):
        config = getattr(self._module, "config", None) # конфиг отсутствует в MultimodalModel, поэтому обращаемся к модулю напрямую
        tied_modules_groups = getattr(config, "tied_modules_groups", []) # [[modules_group]]
        tied_modules_names = {} # {"module_name": (modules_group,)}
        self._tied_modules = defaultdict(list) # {(modules_group,): [module_]}

        for modules_group in tied_modules_groups:
            for module_name in modules_group:
                tied_modules_names[module_name] = tuple(modules_group)

        all_modules = {
            module.super_tensor.name: module.flatten for module in self._module.modules() if isinstance(module, SuperTensorModule)
        }

        for module_name in tied_modules_names:
            if module_name in all_modules:
                modules_group = tied_modules_names[module_name]
                flatten_module = all_modules[module_name]
                self._tied_modules[modules_group].append(flatten_module)
            else:
                raise ValueError(f"There is no SuperTensorModule called {module_name}.")

    @torch.no_grad()
    def _init_tied_modules(self):
        for modules_group in self._tied_modules:
            init_values = self._tied_modules[modules_group][0]
            for flatten_module in self._tied_modules[modules_group][1:]:
                flatten_module.copy_(init_values)

    @torch.no_grad()
    def _sync_tied_modules(self):
        for modules_group in self._tied_modules:
            grad_values = torch.stack([flatten_module.grad for flatten_module in self._tied_modules[modules_group]], dim=0).sum(dim=0)
            for flatten_module in self._tied_modules[modules_group]:
                flatten_module.grad.copy_(grad_values)

    def _setup_meta_info(self):
        self._meta_info = {}
        for buffer_name in ParamBufferPrefix:
            self._meta_info[f"{buffer_name}_parameter_shard_bounaries"] = []  # keeps fragmentation (by modules) of buffer (By shard sizes)
            self._meta_info[f"{buffer_name}_parameter_padded_numels"] = []  # keeps padded numels of each module
            self._meta_info[f"{buffer_name}_shards_fqns"] = []  # keep List[List]. Each list keeps ordered fqns of this module
            self._meta_info[f"{buffer_name}_shards_numels"] = []  # keep List[List]. Each list keeps ordered numels of params of this module
            self._meta_info[f"{buffer_name}_parameter_orig_shapes"] = [] # keep List[List]. Each list keeps ordered orig shapes of params
            self._meta_info[f"{buffer_name}_parameter_pg_size"] = [] # keep List[Tuple]. Each tuple keeps ordered dp and mp size of params
            self._meta_info[f"{buffer_name}_parameter_parallel_type"] = [] # keep List[List]. Each list keeps ordered  type tp split  of params
        self._meta_info['meta_for_hash'] = list(self._meta_info.keys()) + ["is_inference_mode"]

    def _add_info_to_metainfo(self,
                              super_tensor : "SuperTensor",
                              params : List[MetaParam],
                              buffer_name : str):
        if len(params) == 0:
            return

        self._meta_info[f"{buffer_name}_shards_fqns"].append(
            [param.fqn for param in params]
        )
        self._meta_info[f"{buffer_name}_shards_numels"].append([param.numel for param in params])
        self._meta_info[f"{buffer_name}_parameter_orig_shapes"].append(
            [list(param.shape) for param in params]
        )
        padded_numel, shard_size = super_tensor.get_params_metainfo(params)
        self._meta_info[f"{buffer_name}_parameter_padded_numels"].append(padded_numel)
        self._meta_info[f"{buffer_name}_parameter_shard_bounaries"].append(shard_size)

        dp_size = super_tensor._fsdp_process_group.size()
        mp_size = super_tensor._model_parallel_process_group.size() if super_tensor._model_parallel_process_group else 1
        self._meta_info[f"{buffer_name}_parameter_pg_size"].append((dp_size, mp_size))
        self._meta_info[f"{buffer_name}_parameter_parallel_type"].append(
            [param.split_type.value for param in params]
        )

    def _add_global_metainfo(self, special_process_group_fn: Optional[Callable] = None):
        self._meta_info['fsdp_group_size'] = self.pg_manager.default_pg_kit.fsdp_process_group.size()
        if self.pg_manager.default_pg_kit.inter_hybrid_process_group:
            self._meta_info['hybrid_group_count'] = self.pg_manager.default_pg_kit.inter_hybrid_process_group.size()
        else:
            self._meta_info['hybrid_group_count'] = 1
        self._meta_info['hybrid_group_size'] = self.pg_manager.hybrid_group_size

        self._meta_info['is_inference_mode'] = self.is_inference_mode
        self._meta_info['_is_separate_module_trainable'] = self._is_separate_module_trainable
        self._meta_info['_is_main_parameter_trainable'] = self._is_main_parameter_trainable

        self._meta_info["fsdp_group_sizes"] = list()
        for pg in self.pg_manager.pg_mapping:
            self._meta_info["fsdp_group_sizes"].append(pg.size())

        self._meta_info[MetaInfoKeys.CHECKPOINT_VERSION_FIELD] = CHECKPOINT_VERSION

        # add fields for optimizer convertation
        fill_optimizer_meta_info(self, special_process_group_fn=special_process_group_fn)

        logger.info(f"meta_info: {self._meta_info}")

    def _create_separate_module_super_tensor(self, separate_module_params : List[MetaParam]):
        if not self._separate_layer_norm_enabled:
            return

        filter_metaparam = lambda x : [param
                                       for param in separate_module_params
                                       if param.requires_grad == x]

        self._layernorm_supertensor = SuperTensor(
            filter_metaparam(True), # trainable
            filter_metaparam(False), # freezed
            name=f"{ParamBufferPrefix.SEPARATE}_supertensor",
            comm_stream=self._comm_stream,
            fsdp_process_group=self.pg_manager.default_pg_kit.fsdp_process_group,
            intra_node_fsdp_process_group=self.pg_manager.default_pg_kit.intra_node_fsdp_process_group,
            model_parallel_process_group=self.pg_manager.default_pg_kit.model_parallel_process_group,
            inter_hybrid_process_group=self.pg_manager.default_pg_kit.inter_hybrid_process_group,
            all_reduce_grads_across_model_parallel_group=self._all_reduce_grads_across_model_parallel_group,
            zero_stage=self._zero_stage,
            param_dtype=self._param_dtype,
        )
        self._super_tensors.append(self._layernorm_supertensor)
        self._is_separate_module_trainable = self._layernorm_supertensor.is_trainable

        self._add_info_to_metainfo(self._layernorm_supertensor, self._layernorm_supertensor.params, ParamBufferPrefix.SEPARATE)
        self._add_info_to_metainfo(self._layernorm_supertensor,
                                   self._layernorm_supertensor.params_freezed,
                                   ParamBufferPrefix.SEPARATE_FREEZED
                                   if self._is_separate_module_trainable else ParamBufferPrefix.SEPARATE)

        self._layernorm_weight_buffer = ReusableBuffer(
            self._layernorm_supertensor.weight_buffer_size,
            device="cuda",
            dtype=self._param_dtype,
            prepare_stream=self._comm_stream,
            process_stream=self._compute_stream,
        )
        self._layernorm_grad_buffer = (
            ReusableBuffer(
                self._layernorm_supertensor.grad_buffer_size,
                device="cuda",
                dtype=self._param_dtype,
                prepare_stream=self._compute_stream,
                process_stream=self._comm_stream,
                fill_zeros=True,
            )
            if self._is_separate_module_trainable
            else None
        )

        self._all_buffers.append(self._layernorm_weight_buffer)
        if self._is_separate_module_trainable:
            self._all_buffers.append(self._layernorm_grad_buffer)

        if hasattr(self._layer_norm_module_cls, "gigafsdp_normalize_fqns"):
            normalize_fqns = lambda x: list(map(self._layer_norm_module_cls.gigafsdp_normalize_fqns, x))
            self._meta_info[f"{ParamBufferPrefix.SEPARATE}_shards_fqns"] = \
                list(map(normalize_fqns, self._meta_info[f"{ParamBufferPrefix.SEPARATE}_shards_fqns"]))
            self._meta_info[f"{ParamBufferPrefix.SEPARATE_FREEZED}_shards_fqns"] = \
                list(map(normalize_fqns, self._meta_info[f"{ParamBufferPrefix.SEPARATE_FREEZED}_shards_fqns"]))

    def _set_buffers_to_separate_module_super_tensor(self):
        if not self._separate_layer_norm_enabled:
            return

        self._layernorm_supertensor.set_buffers(self._layernorm_weight_buffer, self._layernorm_grad_buffer)
        self._layernorm_supertensor.set_fp32_grad_buffer(self._fp32_grad_buffer)

        self._layernorm_supertensor.set_weight_shard_buffers(
            self._layernorm_parameter_bit16,
            self._layernormfreezed_parameter_bit16 if self._is_separate_module_trainable else self._layernorm_parameter_bit16,
            0,
            0,
            self._layernorm_parameter
        )
        if self._bit32_acc_for_bit16_reduce_scatter:
            self._layernorm_supertensor.enable_bit32_acc_for_bit16_reduce_scatter()

    def _prepare_fp8_scales_for_experts(self):
        """amax -> all-gather -> quantize, ONCE per optimizer step.
        """
        fp8_sts = getattr(self, "_fp8_sts", None)
        if not fp8_sts:
            return
        for st in fp8_sts:
            compute_local_tile_amaxes(st.shard, out=st.local_amaxes)
        dist.all_gather_into_tensor(
            self._expert_gathered_amaxes,
            self._expert_local_amaxes,
            group=self._fp8_group,
            async_op=False,
        )
        for st in fp8_sts:
            st.quantize_shard_fp8()

    def buffer_free(self, buffer):
        if self._zero_stage == 3 or not (buffer.is_weight_buffer and not buffer._owner.is_trainable):
            buffer.free()

    def _get_backward_hook(self):
        def backward_hook():
            if self._separate_layer_norm_enabled and self._layernorm_supertensor._need_manual_backward_hook:
                self._layernorm_supertensor.get_backward_hook()()

            logger.debug(f"for {type(self).__name__} {self._backwards_count=}")

            self._backwards_count = (self._backwards_count + 1) % self._gradient_accumulation_steps
            if self._backwards_count > 0:
                if self._zero_stage > 1 and self._is_separate_module_trainable:
                    self._layernorm_grad_buffer.free()
                return

            for buffer in sorted(self._all_buffers, key=lambda buffer: buffer.last_time_processed):
                self.buffer_free(buffer)

            for super_tensor in self._super_tensors:
                if super_tensor.is_hpz_layer:
                    super_tensor.hpz_inited = False

            comm_wait_event = torch.cuda.Event()
            comm_wait_event.record(self._comm_stream)
            comm_wait_event.wait()

            self._sync_tied_modules()

            if self._is_main_parameter_trainable:
                # for all types of fsdp(ep_fsdp)/tp(ep) groups we need to divide grad by basic fsdp group size,
                # because is equal to DP size
                divide_param_grad(
                    self._main_parameter,
                    self.pg_manager.default_pg_kit,
                    self._gradient_accumulation_steps
                )

            if self._is_separate_module_trainable:
                divide_param_grad(
                    self._layernorm_parameter,
                    self.pg_manager.default_pg_kit,
                    self._gradient_accumulation_steps
                )

        return backward_hook

    def _init_grad_flows_and_set_hook(self):
        flows = []
        for super_tensor in self._super_tensors:
            super_tensor.init_weight_grad_flow()
            logger.debug(f"{super_tensor.name=}, {super_tensor.is_trainable=}, {super_tensor.must_be_gated=}")
            # otherwise backward will be computed throw all layers
            if super_tensor.must_be_gated:
                flows.append(super_tensor.weight_grad_flow)
        flows = GateGradFlow.apply(self._get_backward_hook(), *flows)

        trainable_idx = 0
        for super_tensor in self._super_tensors:
            if super_tensor.must_be_gated:
                super_tensor.weight_grad_flow = flows[trainable_idx]
                trainable_idx += 1

    def copy_master_weight_to_bit16_buffer(self):
        logger.debug(f"{self._is_separate_module_trainable=}, {self._is_main_parameter_trainable=}")
        # Don't need to copy static params
        if self._is_separate_module_trainable:
            self._layernorm_parameter_bit16.data.copy_(self._layernorm_parameter)
        if self._is_main_parameter_trainable:
            self._main_parameter_bit16.data.copy_(self._main_parameter)

    def _get_forward_pre_hook(self):
        def forward_pre_hook(module, input):
            for super_tensor in self._super_tensors:
                if super_tensor._backward_depth != 0:
                    logger.warning(
                        f"{super_tensor.name}: backward_depth=" +
                        f"{super_tensor._backward_depth} at forward entry; a backward " +
                        f"hook did not fire. Resetting to keep the weight-buffer " +
                        f"start/end_processing protocol balanced."
                    )
                    super_tensor._backward_depth = 0

            if self._backwards_count == 0:
                self.copy_master_weight_to_bit16_buffer()
                if self._float8_all_gather_on_experts:
                    self._prepare_fp8_scales_for_experts()

                # Order the bf16 master->shard copy (compute stream) against every
                # all-gather that reads those shards on the comm stream.
                self._comm_stream.wait_stream(torch.cuda.current_stream())

                # In zero - 2 stage in eval mode we do not need to free buffers as they will be used further
                if not (self._zero_stage == 2 and not torch.is_grad_enabled()):
                    for buffer in sorted(self._all_buffers, key=lambda buffer: buffer.last_time_processed):
                        self.buffer_free(buffer)
                    for super_tensor in self._super_tensors:
                        super_tensor.first_reduce_scatter = True
            
            is_last_accumulation_step = (self._gradient_accumulation_steps - self._backwards_count) == 1
            for super_tensor in self._super_tensors:
                super_tensor.set_is_last_accumulation_step(is_last_accumulation_step)

            self._init_grad_flows_and_set_hook()  # set up backward hook
            if self._separate_layer_norm_enabled:
                self._layernorm_supertensor.get_forward_pre_hook()(module, input)
                self._layernorm_supertensor.pre_forward(input)

        return forward_pre_hook

    def __getattr__(self, name: str) -> Union[torch.Tensor, "Module"]:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._module, name)

    def _init_with_order(self):
        gc.collect()
        torch.cuda.empty_cache()

        for super_tensor in self._super_tensors[:self._main_st_count]:
            if super_tensor.first_time_processed is None:
                raise RuntimeError(f"Module {super_tensor.name=} wasn't be called in forward.")

        # sorting super_tensors by forward execution order
        self._super_tensors[:self._main_st_count] = sorted(self._super_tensors[:self._main_st_count],
                                                           key=lambda x: x.first_time_processed)
        for st in self._super_tensors:
            logger.debug(f"Order of modules execution: {st.name=}")

        if self._zero_stage < 3:
            self._weight_buffers = sorted(self._weight_buffers, key=lambda x: x._owner.first_time_processed)

        # for zero_stage=1 we have 1 fwd/bwd buffer per layer (order is unimportant)
        if self._zero_stage > 1:
            self._set_reusable_buffers()
        self._order_inited = True

    def _set_reusable_buffers(self):
        for buffer in self._weight_buffers:
            buffer._free()

        for idx, super_tensor in enumerate(self._super_tensors[:self._main_st_count]):
            super_tensor.remove_buffers()
            super_tensor.set_buffers(
                self._weight_buffers[idx % len(self._weight_buffers)],
                self._grad_buffers[idx % len(self._grad_buffers)],
            )

    @torch.no_grad()
    def fake_forward(self, *args: Any, **kwargs: Any) -> None:
        super().__call__(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._order_inited:
            self.fake_forward(*args, **kwargs)
            self._init_with_order()
        return super().__call__(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the forward pass for the wrapped module."""
        output = self._module(*args, **kwargs)

        if not isinstance(output, torch.Tensor) and not isinstance(output, tuple):
            raise RuntimeError("GigaFSDP supports model output only in the form of tensors or a tuple of tensors. " +
                               f"Not {type(output)}")

        return output

    def clip_grad_norm_(self, max_norm: Union[float, int], norm_type: Union[float, int] = 2.0) -> torch.Tensor:
        assert norm_type in {2.0, 2}

        if self.is_inference_mode:
            return

        def get_total_norm(grad_list : list[torch.Tensor],
                           pg_kit : ProcessGroupKit,
                           divide_first_grad : bool = False):
            grad_norm_list = [
                torch.linalg.vector_norm(grad.detach(), norm_type, dtype=torch.float32)
                for grad in grad_list
            ]

            # divide on tp_size first grad
            if divide_first_grad and pg_kit.model_parallel_process_group:
                grad_norm_list[0] /= pg_kit.model_parallel_process_group.size() ** (1 / norm_type)

            grad_norm = torch.linalg.vector_norm(
                torch.stack(grad_norm_list),
                norm_type,
                dtype=torch.float32,
            )
            total_norm = grad_norm ** norm_type

            dist.all_reduce(total_norm, group=pg_kit.fsdp_process_group)
            if pg_kit.model_parallel_process_group is not None and pg_kit.model_parallel_process_group.size() > 1:
                dist.all_reduce(total_norm, group=pg_kit.model_parallel_process_group)
            return total_norm

        pg_kit_to_grad_list = defaultdict(list)
        if self._is_separate_module_trainable:
            # layernorm always on first place
            pg_kit_to_grad_list[self.pg_manager.default_pg_kit].append(self._layernorm_parameter.grad)

        if self.pg_manager.is_trivial_manager: # only one fsdp process group
            if self._is_main_parameter_trainable:
                pg_kit_to_grad_list[self.pg_manager.default_pg_kit].append(self._main_parameter.grad)
        else: # multiple fsdp process groups
            for module in self._module.modules():
                if isinstance(module, SuperTensorModule) and module.super_tensor.is_trainable:
                    pg_kit_to_grad_list[module.pg_kit].append(module.flatten.grad)

        total_norm = 0
        for pg_kit, grad_list in pg_kit_to_grad_list.items():
            # layernorms have grads and this group is default fsdp group
            divide_first_grad=self._is_separate_module_trainable and pg_kit is self.pg_manager.default_pg_kit
            total_norm += get_total_norm(grad_list,
                                         pg_kit,
                                         divide_first_grad=divide_first_grad)

        total_norm = total_norm ** (1 / norm_type)
        total_norm_coef = max_norm / (total_norm + 1e-6)
        total_norm_coef = torch.clamp(total_norm_coef, max=1.0)
        if self._is_separate_module_trainable:
            self._layernorm_parameter.grad.mul_(total_norm_coef)
        if self._is_main_parameter_trainable:
            self._main_parameter.grad.mul_(total_norm_coef)
        return total_norm

    def local_state_dict(self):
        state_dict = super().state_dict()
        state_dict["meta"] = self._meta_info
        return state_dict

    def _restore_fp8_nan_carrier(self):
        """Re-poison the shared bf16 scratch that fp8 SuperTensors hand to autograd.
        """
        scratch = getattr(self, "_expert_bf16_scratch", None)
        if scratch is not None:
            scratch.fill_(float("nan"))

    def state_dict(self):
        state_dict = OrderedDict()
        try:
            for super_tensor in self._super_tensors:
                super_tensor.collect_module()
                if self.pg_manager.default_pg_kit.fsdp_process_group.rank() != 0:
                    continue
                for param in super_tensor.params + super_tensor.params_freezed:
                    data = getattr(param.parent_module, param.name)
                    state_dict[f"{param.parent_module_name}.{param.name}"] = data.detach().cpu()
        finally:
            self._restore_fp8_nan_carrier()
        return state_dict

    def _get_load_state_dict_pre_hook(self):
        # Deprecated hook. Use utils.py::load_gigafsdp_model_state_dict instead.
        def load_state_dict_pre_hook(module, state_dict, *args, **kwargs):
            state_dict.pop("meta", None)
            assert torch.__future__.get_swap_module_params_on_conversion() == False, \
                "Can't use swap_tensors while load GigaFSDP state_dict"
            module.has_ema_flag = any(key.endswith(EMA_SUFFIX) for key in state_dict.keys())
        return load_state_dict_pre_hook

    def _get_load_state_dict_post_hook(self):
        def load_state_dict_post_hook(module, state_dict, *args, **kwargs):
            module.copy_master_weight_to_bit16_buffer()
        return load_state_dict_post_hook

    def zero_grad(self, *args, **kwargs) -> None:
        set_to_none = False
        if args:
            assert len(args) == 1 and not kwargs
            set_to_none = args[0]
        elif kwargs:
            assert len(kwargs) == 1 and (key := "set_to_none") in kwargs and not args
            set_to_none = kwargs[key]
        if set_to_none:
            raise ValueError("GigaFSDP doesn't support zero_grad(set_to_none=True).")
        super().zero_grad(set_to_none)

    def get_memory_stats(self, measurement='GB'):
        def get_stats(tensor, measurement=''):
            divider = {'GB': 1024**3, 'MB': 1024**2, 'KB': 1024}.get(measurement, 1)
            return tensor.numel() * tensor.element_size() / divider

        memory_stats = defaultdict(float)
        optimizer_memory = 0
        for buffer in self._weight_buffers:
            memory_stats['weight_buffers'] += get_stats(buffer._buffer, measurement)

        for buffer in self._grad_buffers:
            if buffer:
                memory_stats['grad_buffers'] += get_stats(buffer._buffer, measurement)

        seen_ids = set()
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                id_ = id(attr.untyped_storage())
                if id_ not in seen_ids:
                    memory_stats[attr_name] += get_stats(attr, measurement)
                    seen_ids.add(id_)
                    if attr.requires_grad:
                        optimizer_memory += 2 * get_stats(attr, measurement)
                    if attr.grad is not None:
                        memory_stats[f'{attr_name}.grad'] += get_stats(attr.grad, measurement)
            elif isinstance(attr, ReusableBuffer):
                memory_stats[attr_name] += get_stats(attr._buffer, measurement)

        msg = "\n" + "="*40 + "\n"
        summary_memory = 0
        for name, value in memory_stats.items():
            summary_memory += value
            if value < 0.001:
                continue
            msg += f"{name:<30}: {value:.3f} {measurement}\n"

        msg += "-"*40 + "\n"
        msg += f"{'summary memory':<30}: {summary_memory:.3f} {measurement}\n"
        msg += "-"*40 + "\n"
        msg += f"{'optimizer memory':<30}: {optimizer_memory:.3f} {measurement}\n"
        msg += "="*40 + "\n"

        return msg

    @property
    def communication_stream(self):
        return self._comm_stream
    
    @property
    def compute_stream(self):
        return self._compute_stream
    
    @staticmethod
    def _get_pg_kit_for_module(
        module: nn.Module, 
        pg_manager: ProcessGroupManager, 
        special_process_group_fn: Optional[Callable],
    ) -> ProcessGroupKit:
        if special_process_group_fn is not None and (fsdp_tp_groups := special_process_group_fn(module)) is not None:
            assert len(fsdp_tp_groups) == 2, "special_process_group_fn must return a tuple of two ProcessGroups"
            pg_manager.add_pg_kit(*fsdp_tp_groups)
            pg_kit = pg_manager.get_pg_kit(fsdp_tp_groups[0])
        else:
            pg_kit = pg_manager.default_pg_kit
        return pg_kit


class ReusableBuffer:
    def __init__(self, size, dtype, device, prepare_stream, process_stream, fill_zeros=False):
        self._buffer = (
            torch.zeros(size, dtype=dtype, device=device)
            if fill_zeros
            else torch.empty(size, dtype=dtype, device=device)
        )
        self._owner = None

        self._prepared_event = None
        self._released_event = None

        self._active_child = None
        self._prev_child = None

        self.last_time_processed = 0

        self._main_prepare_stream = prepare_stream
        self._prepare_stream = prepare_stream
        self._process_stream = process_stream

    def set_prepare_stream(self, stream):
        self._prepare_stream = stream

    def unset_prepare_stream(self):
        self._prepare_stream = self._main_prepare_stream

    def get_prefix(self, real_size, padded_size, name=""):
        return ReusableBufferView(self, real_size, padded_size, name)

    def _set_owner(self, buffer_view, super_tensor):
        assert self._owner is None
        assert self._active_child is None
        self._owner = super_tensor
        self._active_child = buffer_view

    def is_gained(self, super_tensor):
        return self._owner == super_tensor

    # only in start_prepare we can change owner
    def start_prepare(self, buffer_view, super_tensor):
        if self.is_gained(super_tensor):
            return

        self._free(super_tensor)
        self._prepared_event = None
        self._released_event = None
        self._set_owner(buffer_view, super_tensor)

    def end_prepare(self, super_tensor):
        assert self._owner == super_tensor, f"{self._owner} != {super_tensor}"
        self._prepared_event = torch.cuda.Event()
        self._prepared_event.record(self._prepare_stream)

    def start_processing(self, super_tensor):
        assert self._owner == super_tensor, f"{self._owner} != {super_tensor}"
        assert self._prepared_event is not None
        self._prepared_event.wait(self._process_stream)
        self._released_event = None

    def end_processing(self, super_tensor):
        assert self._owner == super_tensor, f"{self._owner} != {super_tensor}"
        self._released_event = torch.cuda.Event()
        self._released_event.record(self._process_stream)
        self.last_time_processed = time.time()

    def _free(self, new_owner : Optional["SuperTensor"] = None):
        new_owner_name = new_owner.name if new_owner else None
        # we use self._prepare_stream because we want to free only after end_processing
        with torch.cuda.stream(self._prepare_stream):
            if self._owner:
                for hook in self._active_child._free_hooks:
                    # the hook implies a delayed call of something that will work in the process_stream
                    hook(self._process_stream)
                if self._released_event is not None:
                    self._released_event.wait()
                else:
                    # An all-gather nobody consumed: a module called twice in one
                    # forward prefetches whatever followed its first call, which
                    # may not run again. Nothing read the weights, so hand the
                    # buffer on -- the module re-gathers when it does run. Only
                    # the write order still has to hold.
                    assert self._active_child._state is not ReusableBufferViewState.IN_PROCESS, \
                        f"SuperTensor {self._owner.name=} didn't call end_processing. {new_owner_name} can't free it."
                    if self._prepared_event is not None:
                        self._prepared_event.wait()
                self._owner.released_event = None
                self._owner.gathered = False
                self._owner = None
                self._prev_child = self._active_child
                self._active_child = None
        self._state = ReusableBufferViewState.NOT_READY

    def free(self):
        if self._active_child is not None:
            self._active_child.free()

    @property
    def is_weight_buffer(self):
        return self._owner is not None and self._owner.weight_reusable_buffer.reusable_buffer is self


class ReusableBufferView:
    def __init__(self, reusable_buffer, real_size, padded_size, name=""):
        self._name = name
        self._real_size = real_size
        self._size = padded_size
        self.reusable_buffer = reusable_buffer
        self._buffer_view = reusable_buffer._buffer.narrow(0, 0, padded_size)
        self._free_hooks = []
        self._state = ReusableBufferViewState.NOT_READY

    def get(self):
        return self._buffer_view

    def set_prepare_stream(self, stream):
        self.reusable_buffer.set_prepare_stream(stream)

    def unset_prepare_stream(self):
        self.reusable_buffer.unset_prepare_stream()

    # method to understand if we have zero grads
    def is_super_tensor_superset_of_previous_owner(self):
        if self.reusable_buffer._prev_child is None:
            return False
        return self._real_size >= self.reusable_buffer._prev_child._real_size

    def start_prepare(self, super_tensor):
        if not self.reusable_buffer.is_gained(super_tensor):
            self.reusable_buffer.start_prepare(self, super_tensor)
            self._state = ReusableBufferViewState.NOT_READY

    def end_prepare(self, super_tensor):
        self.reusable_buffer.end_prepare(super_tensor)
        self._state = ReusableBufferViewState.READY

    def start_processing(self, super_tensor):
        assert self._state == ReusableBufferViewState.READY, f"real state is {self._state} for {self._name=}"
        self.reusable_buffer.start_processing(super_tensor)
        self._state = ReusableBufferViewState.IN_PROCESS

    def end_processing(self, super_tensor):
        assert self._state == ReusableBufferViewState.IN_PROCESS, f"real state is {self._state} for {self._name=}"
        self.last_time_gained = time.time()
        self.reusable_buffer.end_processing(super_tensor)
        self._state = ReusableBufferViewState.READY

    def is_gained(self, super_tensor):
        return self.reusable_buffer.is_gained(super_tensor)

    def free(self):
        self.reusable_buffer._free()
        self._state = ReusableBufferViewState.NOT_READY

    def register_free_hook(self, hook):
        self._free_hooks.append(hook)


# this function guaranties that gate gradient will be computed before input gradient
class GateGradFlow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hook, *inputs):
        ctx.set_materialize_grads(False)
        ctx.hook = hook
        return inputs

    @staticmethod
    def backward(ctx, *grads):
        ctx.hook()
        return None, *grads


@torch._dynamo.allow_in_graph
class TyingGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fake_x):
        ctx.save_for_backward(fake_x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        fake_x, = ctx.saved_tensors
        return grad_output, fake_x


class SuperTensor:
    def __init__(
        self,
        params,
        params_freezed,
        comm_stream,
        name,
        fsdp_process_group: torch.distributed.ProcessGroup,
        intra_node_fsdp_process_group: Optional[torch.distributed.ProcessGroup] = None,
        model_parallel_process_group: Optional[torch.distributed.ProcessGroup] = None,
        inter_hybrid_process_group: Optional[torch.distributed.ProcessGroup] = None,
        all_reduce_grads_across_model_parallel_group=False,
        is_hpz_layer=False,
        hpz_all_gather_stream=None,
        exec_order_tracker: Optional[ExecutionOrderTracker] = None,
        is_ignored_module_trainable: bool = False,
        zero_stage: int = 3,
        alignment_value: Optional[int] = None,
        fp8_all_gather_selected: bool = False,
        param_dtype: torch.dtype = torch.bfloat16,
    ):
        assert len(params + params_freezed) > 0, f"There are no params at layer {name=}. Can't wrap it in STM."
        assert all(isinstance(param, MetaParam) for param in params + params_freezed)

        self.name = name
        self.is_trainable = len(params) > 0 # is any trainable param
        self.is_ignored_module_trainable = is_ignored_module_trainable
        self.comm_stream = comm_stream
        self.exec_order_tracker = exec_order_tracker

        self._fsdp_process_group = fsdp_process_group
        self.rank = fsdp_process_group.rank()
        self.world_size = fsdp_process_group.size()
        self.is_hpz_layer = is_hpz_layer

        self.params_shapes = [p.shape for p in params]
        self.params_shapes_freezed = [p.shape for p in params_freezed]

        self.params : list[MetaParam] = params
        self.params_freezed : list[Optional[MetaParam]] = params_freezed
        self.shard_offset : Optional[int] = None
        self.shard_offset_freezed : Optional[int] = None

        self._zero_stage = zero_stage
        self._alignment_value = alignment_value
        self._param_dtype = param_dtype

        self.is_fp8: bool = False
        self.fp8_geom: List[Fp8ParamGeom] = []
        self.shard_fp8: Optional[torch.Tensor] = None
        self.local_amaxes: Optional[torch.Tensor] = None
        self.gathered_amaxes: Optional[torch.Tensor] = None
        self.block_scales: Optional[List[torch.Tensor]] = None
        self.scale_block_offset: Optional[int] = None
        self.packed_scale_stride: Optional[int] = None
        self.local_scale_blocks: Optional[int] = None
        self._fp8_weights_quantized: Optional[dict] = None
        self.bf16_weight_scratch: Optional[torch.Tensor] = None  # static BF16 scratch for _materialize_params
        self._backward_depth: int = 0

        (   self.total_numel,
            self.padded_numel,
            self.shard,
            self.shard_size,
            self.param_fqn_to_offset,
            self.param_fqn_to_shape
        ) = self._init_shard(self.params)

        (   self.total_numel_freezed,
            self.padded_numel_freezed,
            self.shard_freezed,
            self.shard_size_freezed,
            self.param_fqn_to_offset_freezed,
            self.param_fqn_to_shape_freezed
        ) = self._init_shard(self.params_freezed, dtype=self._param_dtype)

        self.is_fp8 = self._init_fp8_eligibility(fp8_all_gather_selected, is_hpz_layer)

        if self.is_hpz_layer:
            assert intra_node_fsdp_process_group is not None
            self.hpz_rank = intra_node_fsdp_process_group.rank()
            self.hpz_world_size = intra_node_fsdp_process_group.size()
            self.hpz_group = intra_node_fsdp_process_group
            self.hpz_all_gather_stream = hpz_all_gather_stream
            self.hpz_shard_size = (self.padded_numel + self.padded_numel_freezed) // self.hpz_world_size
            self.hpz_inited = False

        self.weight_reusable_buffer = None
        self.grad_reusable_buffer = None
        self.first_reduce_scatter = True  # For ZeRO-2,3

        # Set by metric_sampler's adapter; called with the full unsharded local gradient
        # buffer right before it is reduce-scattered across `_fsdp_process_group`.
        self.pre_reduce_observer = None

        self.bit32_acc_for_bit16_reduce_scatter = False

        self.all_reduce_grads_across_model_parallel_group = all_reduce_grads_across_model_parallel_group
        self._model_parallel_process_group = model_parallel_process_group

        self._inter_hybrid_process_group = inter_hybrid_process_group
        self._need_manual_backward_hook: bool = False
        self.first_time_processed: Optional[int] = None

        self._is_last_accumulation_step = False

    def get_param_fqns_to_shapes(self):
        return {'frozen': self.param_fqn_to_shape_freezed, 'active': self.param_fqn_to_shape}
    
    def get_param_fqns_to_offsets(self):
        return {'frozen': self.param_fqn_to_offset_freezed, 'active': self.param_fqn_to_offset}

    @property
    def alignment_value(self) -> int:
        return self._alignment_value

    def _get_alignment(self) -> int:
        if self._alignment_value is not None:
            return self.world_size * self._alignment_value
        return self.world_size

    def _init_fp8_eligibility(self, fp8_all_gather_selected: bool, is_hpz_layer: bool) -> bool:
        """Decide once whether this SuperTensor is all-gathered in fp8, and build its geometry.
        """
        if not fp8_all_gather_selected:
            return False

        expert_fqns = [p.fqn for p in self.params if is_expert_param(p.fqn)]
        if not expert_fqns:
            logger.info(f"[fp8] {self.name}: selected but holds no expert weights; using bf16 all-gather")
            return False

        other_fqns = [p.fqn for p in self.params if not is_expert_param(p.fqn)]
        assert not other_fqns, (
            f"[fp8] {self.name}: fp8 all-gather requires the SuperTensor to contain ONLY expert "
            f"weights, but it also holds {other_fqns} alongside {expert_fqns}. "
            f"Exclude this module from fp8 all-gather (float8_all_gather_skip_patterns) "
            f"or wrap the experts in their own module."
        )
        assert self.padded_numel_freezed == 0, (
            f"[fp8] {self.name}: frozen params are not supported with fp8 all-gather "
            f"({self.padded_numel_freezed=})."
        )
        assert self._zero_stage == 3, (
            f"[fp8] {self.name}: fp8 all-gather requires zero_stage == 3, got {self._zero_stage}."
        )
        assert not is_hpz_layer, f"[fp8] {self.name}: HPZ layers cannot use fp8 all-gather."
        assert self.shard_size % FP8_TILE_SIZE == 0, (
            f"[fp8] {self.name}: shard_size={self.shard_size} is not a multiple of {FP8_TILE_SIZE}; "
            f"fp8 all-gather requires alignment_value=128."
        )

        geom: List[Fp8ParamGeom] = []
        for p in self.params:
            shape = self.param_fqn_to_shape[p.fqn]
            assert fp8_param_shape_is_supported(shape), (
                f"[fp8] {self.name}: param {p.fqn} has shape {tuple(shape)}; fp8 all-gather supports "
                f"2-D [N, K] or 3-D [E, N, K] with both trailing dims divisible by {FP8_TILE_SIZE}."
            )
            elem_offset = self.param_fqn_to_offset[p.fqn]
            assert elem_offset % FP8_TILE_SIZE == 0, (
                f"[fp8] {self.name}: param {p.fqn} starts at element {elem_offset}, "
                f"not a multiple of {FP8_TILE_SIZE}."
            )
            geom.append(Fp8ParamGeom(
                fqn=p.fqn,
                elem_offset=elem_offset,
                tile_offset=elem_offset // FP8_TILE_SIZE,
                n_experts=shape[0] if len(shape) == 3 else 1,
                n_row_elem=shape[-2],
                n_cols=shape[-1],
            ))

        self.fp8_geom = geom
        # TODO (fedorovgv): check this
        # Mark the owning modules so every consumer can detect that these weights
        # are GigaFSDP-managed fp8 and that the bf16 tensors are a NaN carrier.
        for p in self.params:
            managed = getattr(p.parent_module, "_giga_fp8_managed_params", None) or set()
            p.parent_module._giga_fp8_managed_params = managed | {p.name}
            setattr(p.parent_module, f"_{p.name}_quantized", None)
        logger.info(f"[fp8] {self.name}: fp8 all-gather enabled for {[g.fqn for g in geom]}")
        return True

    def _init_shard(self, params : list[MetaParam], dtype: Optional[torch.dtype] = None):
        param_fqn_to_offset = {}
        param_fqn_to_shape = {}
        total_numel = 0

        for p in params:
            fqn = p.fqn

            param_fqn_to_offset[fqn] = total_numel
            param_fqn_to_shape[fqn] = tuple(p.shape)
            
            total_numel += p.numel
        
        padded_numel = total_numel

        if total_numel == 0:
            return total_numel, padded_numel, None, 0, param_fqn_to_offset, param_fqn_to_shape

        alignment = self._get_alignment()
        if padded_numel % alignment != 0:
            padded_numel += alignment - padded_numel % alignment

        flatten_buffer = torch.cat(
            [p.param.view(-1).to(dtype or p.param.dtype) for p in params]
            + [
                torch.zeros(
                    padded_numel - total_numel,
                    dtype=dtype or params[0].param.dtype,
                    device=params[0].param.device
                )
            ]
        )

        for param in params:
            param.clear_param()
        torch.cuda.empty_cache()

        shard_size = padded_numel // self.world_size
        shard = flatten_buffer.narrow(0, shard_size * self.rank, shard_size).clone()

        del flatten_buffer
        torch.cuda.empty_cache()

        return total_numel, padded_numel, shard, shard_size, param_fqn_to_offset, param_fqn_to_shape

    def get_params_metainfo(self, params: list[MetaParam]) -> Tuple[int, int]:
        if params is self.params:
            return self.padded_numel, self.shard_size
        elif params is self.params_freezed:
            return self.padded_numel_freezed, self.shard_size_freezed
        else:
            raise ValueError(f"Unknown params list {params} for this supertensor")

    @property
    def fp8_tail_size(self) -> int:
        """Size, in weight-buffer (param_dtype) elements, of the fp8 byte tail.
        """
        assert self.is_fp8
        elem_size = torch.empty((), dtype=self._param_dtype).element_size()
        return (self.padded_numel + elem_size - 1) // elem_size

    @property
    def weight_buffer_size(self):
        if self.is_fp8:
            return self.fp8_tail_size
        return self.padded_numel + self.padded_numel_freezed

    @property
    def grad_buffer_size(self):
        return self.padded_numel

    @property
    def debug_msg(self):
        return f"for {type(self).__name__} {self.name=}"

    @property
    def must_be_gated(self):
        return self.is_trainable or self.is_ignored_module_trainable

    def set_is_last_accumulation_step(self, is_last_accumulation_step):
        self._is_last_accumulation_step = is_last_accumulation_step

    def set_buffers(self, weight_reusable_buffer, grad_reusable_buffer):
        assert self.weight_reusable_buffer is None, "attempt to set buffer for initialized supertensor"
        if self.is_fp8:
            real_size = self.fp8_tail_size
        else:
            real_size = self.total_numel_freezed if not self.is_trainable \
                        else self.padded_numel + self.total_numel_freezed

        self.weight_reusable_buffer = weight_reusable_buffer.get_prefix(
            real_size, self.weight_buffer_size, f"{self.name}_weight"
        )
        if grad_reusable_buffer is not None and self.is_trainable:
            self.grad_reusable_buffer = grad_reusable_buffer.get_prefix(
                self.total_numel, self.grad_buffer_size, f"{self.name}_grad"
            )
            self.grad_reusable_buffer.register_free_hook(lambda stream: self.reduce_scatter_grad(stream))

    @property
    def fp8_active(self) -> bool:
        return self.is_fp8 and self.shard_fp8 is not None

    def init_fp8_expert_state(
        self,
        local_amaxes: torch.Tensor,
        gathered_amaxes: torch.Tensor,
        scale_offset: int,
        packed_scale_stride: int,
        bf16_scratch: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Allocate the persistent fp8 buffers for this SuperTensor.
        """
        assert self.is_fp8, f"{self.name}: init_fp8_expert_state on a non-fp8 SuperTensor"
        self.shard_fp8 = torch.zeros(self.shard_size, dtype=torch.float8_e4m3fn, device=device)
        # One 128x128 block-scale grid per parameter, laid out [E, N//128, K//128]
        # (non-contraction dim first) to match the grouped-GEMM contract.
        self.block_scales = [
            torch.empty(
                (g.n_experts, g.n_row_elem // FP8_TILE_SIZE, g.n_cols // FP8_TILE_SIZE),
                dtype=torch.float32, device=device,
            )
            for g in self.fp8_geom
        ]
        self.local_amaxes = local_amaxes
        self.gathered_amaxes = gathered_amaxes
        self.scale_block_offset = scale_offset
        self.packed_scale_stride = packed_scale_stride
        self.local_scale_blocks = self.shard_size // FP8_TILE_SIZE
        self.bf16_weight_scratch = bf16_scratch

    def remove_buffers(self):
        self.weight_reusable_buffer = None
        self.grad_reusable_buffer = None

    def set_fp32_grad_buffer(self, reusable_fp32_grad_buffer):
        if reusable_fp32_grad_buffer is not None and self.is_trainable:
            self.reusable_fp32_grad_buffer = reusable_fp32_grad_buffer.get_prefix(
                self.total_numel, self.grad_buffer_size, f"{self.name}_fp32_buffer"
            )
        else:
            self.reusable_fp32_grad_buffer = None

    def set_weight_shard_buffers(
            self,
            weight_shard_buffer : torch.Tensor,
            weight_freezed_shard_buffer : Optional[torch.Tensor],
            offset : int,
            offset_freezed : int,
            fp32_shard_buffer : nn.Parameter
        ):
        if self.params:
            # main_parameter
            old_shard = self.shard.data
            self.shard_offset = offset
            self.shard.data = weight_shard_buffer.narrow(0, offset, self.shard_size)
            self.shard.data.copy_(old_shard)

            # main_parameter_fp32
            fp32_shard_buffer.data[offset : offset + self.shard_size] = old_shard
            assert fp32_shard_buffer.grad is not None, \
                f"f{self.is_trainable=} for supertensor {self.name=}, but {fp32_shard_buffer.grad=}"
            self.fp32_grad_shard = fp32_shard_buffer.grad.narrow(0, offset, self.shard_size)

            del old_shard

        # freezed_parameter
        if self.params_freezed:
            assert weight_freezed_shard_buffer is not None, \
                f"{len(self.params_freezed)=} for supertensor {self.name=}, but {weight_freezed_shard_buffer=}"

            old_shard = self.shard_freezed.data
            self.shard_offset_freezed = offset_freezed
            self.shard_freezed.data = weight_freezed_shard_buffer.narrow(0, offset_freezed, self.shard_size_freezed)
            self.shard_freezed.data.copy_(old_shard)
            del old_shard

        if self.is_hpz_layer:
            if self.shard is not None:
                device, dtype = self.shard.device, self.shard.dtype
            else:
                device, dtype = self.shard_freezed.device, self.shard_freezed.dtype
            self.hpz_buffer = torch.empty(self.hpz_shard_size, device=device, dtype=dtype)
            assert self.weight_reusable_buffer.get().shape[0] % self.hpz_buffer.shape[0] == 0

    def enable_bit32_acc_for_bit16_reduce_scatter(self):
        self.bit32_acc_for_bit16_reduce_scatter = True

    def _all_gather(self, output_buffer: torch.Tensor):
        is_double_allgather = self.padded_numel != 0 and self.need_second_allgather
        ctx_mgr = dist._coalescing_manager(group=self._fsdp_process_group) \
                  if is_double_allgather else contextlib.nullcontext()

        with ctx_mgr:
            if self.padded_numel:
                dist.all_gather_into_tensor(
                    output_buffer.narrow(0, 0, self.padded_numel),
                    self.shard,
                    group=self._fsdp_process_group, async_op=False)

            if self.need_second_allgather:
                dist.all_gather_into_tensor(
                    output_buffer.narrow(0, self.padded_numel, self.padded_numel_freezed),
                    self.shard_freezed,
                    group=self._fsdp_process_group, async_op=False)

    def quantize_shard_fp8(self):
        """Quantize this rank's bf16 shard into ``shard_fp8`` using the gathered amaxes.
        """
        rank, world_size = self.rank, self.world_size

        assert self.scale_block_offset is not None
        assert self.packed_scale_stride is not None
        assert self.local_scale_blocks is not None
        assert self.block_scales is not None

        for geom, block_scale in zip(self.fp8_geom, self.block_scales):
            fold_and_quantize_param(
                gathered_scales=self.gathered_amaxes,
                scale_block_offset=self.scale_block_offset,
                packed_scale_stride=self.packed_scale_stride,
                local_scale_blocks=self.local_scale_blocks,
                shard=self.shard,
                out_fp8=self.shard_fp8,
                out_block_scale=block_scale,
                n_experts=geom.n_experts,
                n_row_elem=geom.n_row_elem,
                n_cols=geom.n_cols,
                elem_offset=geom.elem_offset,
                tile_offset=geom.tile_offset,
                rank=rank,
                world_size=world_size,
            )

    def _fp8_all_gather(self):
        assert self.weight_reusable_buffer is not None
        assert not self.need_second_allgather, (
            f"[fp8] {self.name}: frozen params are not supported in FP8 buffer mode."
        )
        buf = self.weight_reusable_buffer.get()
        dist.all_gather_into_tensor(
            buf.view(torch.uint8).narrow(0, 0, self.padded_numel),
            self.shard_fp8.view(torch.uint8),
            group=self._fsdp_process_group,
            async_op=False,
        )
        gathered_fp8 = buf.view(torch.float8_e4m3fn).narrow(0, 0, self.padded_numel)
        self._build_fp8_quantized_views(gathered_fp8)

    def _build_fp8_quantized_views(self, gathered_fp8: torch.Tensor) -> None:
        """Publish one ``(data, scale)`` pair per fp8 parameter.
        """
        assert self.block_scales is not None
        quantized = {}
        for geom, block_scale in zip(self.fp8_geom, self.block_scales):
            numel = geom.n_experts * geom.n_row_elem * geom.n_cols
            weight_flat = gathered_fp8.narrow(0, geom.elem_offset, numel)
            if geom.n_experts == 1:
                weight_view = weight_flat.view(geom.n_row_elem, geom.n_cols)
                scale_view = block_scale.squeeze(0)
            else:
                weight_view = weight_flat.view(geom.n_experts, geom.n_row_elem, geom.n_cols)
                scale_view = block_scale.view(block_scale.shape)
            quantized[geom.fqn] = (weight_view, scale_view)
        self._fp8_weights_quantized = quantized
        self._publish_fp8_quantized_attrs()

    def _publish_fp8_quantized_attrs(self) -> None:
        """Expose the quantized weights on the owning modules as ``_<name>_quantized``.
        """
        assert self._fp8_weights_quantized is not None
        for param in self.params:
            value = self._fp8_weights_quantized.get(param.fqn)
            assert value is not None, f"[fp8] {self.name}: no quantized views for {param.fqn}"
            setattr(param.parent_module, f"_{param.name}_quantized", value)

    def _clear_fp8_quantized_attrs(self) -> None:
        for param in self.params:
            setattr(param.parent_module, f"_{param.name}_quantized", None)

    def all_gather(self, comm_stream):
        if self.weight_reusable_buffer.is_gained(self):
            # The buffer still holds THIS SuperTensor's data, so no collective is
            # needed -- but the quantized views and the handoff attrs are cleared at
            # the end of every accumulation cycle, and an activation-recompute
            # forward lands here afterwards.  REBUILD them from the buffer rather
            # than re-publishing a cache: is_gained() means the bytes in there are
            # ours, so the views are always reconstructible, whereas the cache may
            # legitimately be None.
            if self.fp8_active:
                self._build_fp8_quantized_views(
                    self.weight_reusable_buffer.get()
                        .view(torch.float8_e4m3fn)
                        .narrow(0, 0, self.padded_numel)
                )
            return
        gather_from_hpz = self.is_hpz_layer and self.hpz_inited
        all_gather_stream = comm_stream if not gather_from_hpz else self.hpz_all_gather_stream
        with torch.cuda.stream(all_gather_stream):
            self.weight_reusable_buffer.set_prepare_stream(all_gather_stream)
            self.weight_reusable_buffer.start_prepare(self)
            output_buffer = self.weight_reusable_buffer.get()

            if gather_from_hpz:
                dist.all_gather_into_tensor(
                    output_buffer,
                    self.hpz_buffer,
                    group=self.hpz_group,
                    async_op=False,
                )
            elif self.fp8_active:
                self._fp8_all_gather()
            else:
                if self.is_fp8:
                    self._fp8_weights_quantized = None
                    self._clear_fp8_quantized_attrs()
                self._all_gather(output_buffer)

            self.weight_reusable_buffer.end_prepare(self)
            self.weight_reusable_buffer.unset_prepare_stream()

    @property
    def need_second_allgather(self):
        return self.padded_numel_freezed != 0 and (self._zero_stage == 3 or self.first_time_processed is None)

    def get_buffer_to_reduce_scatter(self, stream):
        if self.reusable_fp32_grad_buffer is None:
            return self.grad_reusable_buffer
        with torch.cuda.stream(stream):
            self.grad_reusable_buffer.start_processing(self)
            self.reusable_fp32_grad_buffer.start_prepare(self)
            self.reusable_fp32_grad_buffer.get().copy_(self.grad_reusable_buffer.get())
            self.reusable_fp32_grad_buffer.end_prepare(self)
            self.grad_reusable_buffer.end_processing(self)
        return self.reusable_fp32_grad_buffer

    def reduce_scatter_grad(self, comm_stream):
        with torch.cuda.stream(comm_stream):
            reduce_grad_buffer = self.get_buffer_to_reduce_scatter(comm_stream)
            reduce_grad_buffer.start_processing(self)
            if _grad_nan_armed:
                accumulate_local_grad_nan(self, reduce_grad_buffer.get())

            # If it's the first call of reduce_scatter, fp32_grad_shard is zero-filled, so it can be used as the output
            reduce_in_master_grad = self.first_reduce_scatter and self.reusable_fp32_grad_buffer is not None

            if reduce_in_master_grad:
                target_tensor = self.fp32_grad_shard
            else:
                target_tensor = reduce_grad_buffer.get().narrow(
                    0, self.shard_size * self._fsdp_process_group.rank(), self.shard_size
                )
            kwargs = dict(
                group=self._fsdp_process_group,
                async_op=False,
            )
            if self.bit32_acc_for_bit16_reduce_scatter and target_tensor.dtype == torch.bfloat16:
                # Only bfloat16 -> float32 ReduceScatter is currently supported.
                kwargs["acc_type"] = torch.float32
            if self.pre_reduce_observer is not None:
                self.pre_reduce_observer(reduce_grad_buffer.get())
            torch.distributed.reduce_scatter_tensor(
                target_tensor,
                reduce_grad_buffer.get(),
                **kwargs,
            )

            if self._inter_hybrid_process_group:
                torch.distributed.all_reduce(target_tensor, group=self._inter_hybrid_process_group)

            if self.all_reduce_grads_across_model_parallel_group:
                torch.distributed.all_reduce(target_tensor, group=self._model_parallel_process_group)

            if not reduce_in_master_grad:
                self.fp32_grad_shard.add_(target_tensor)

            reduce_grad_buffer.end_processing(self)
            self.first_reduce_scatter = False
            for param in self.params:
                param.grad_accumulation = False

    def _split_weight_buffer(self,
                             weight_buffer : torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        trainable = weight_buffer.narrow(0, 0, self.padded_numel)
        freezed = weight_buffer.narrow(0, self.padded_numel, self.padded_numel_freezed)
        return trainable, freezed

    def init_weight_grad_flow(self):
        if self.fp8_active:
            # FP8 primary mode: weight_reusable_buffer holds FP8 data, so we cannot
            # use it for the autograd BF16 weight view.  Use the shared static BF16
            # scratch buffer instead.  The content is never read — the actual GEMM
            # consumes the quantized (data, scale) pairs published by the all-gather.
            assert self.bf16_weight_scratch is not None, \
                f"{self.name}: bf16_weight_scratch not set for fp8 SuperTensor"
            trainable_weight = self.bf16_weight_scratch.narrow(0, 0, self.padded_numel)
            # fp8 STs have padded_numel_freezed == 0 (enforced by _init_fp8_eligibility)
            freezed_weight = self.bf16_weight_scratch.narrow(0, self.padded_numel, self.padded_numel_freezed)
            dtype = trainable_weight.dtype
            device = trainable_weight.device
        else:
            assert self.weight_reusable_buffer is not None
            weight_reusable_buffer_ = self.weight_reusable_buffer.get()
            trainable_weight, freezed_weight = self._split_weight_buffer(weight_reusable_buffer_)
            dtype = weight_reusable_buffer_.dtype
            device = weight_reusable_buffer_.device

        self.weight_grad_flow = torch.empty(
            0,
            dtype=dtype,
            device=device,
            requires_grad=self.is_trainable,
        )
        self.weight_grad_flow.data = trainable_weight
        if self.grad_reusable_buffer is not None:
            self.weight_grad_flow.grad = self.grad_reusable_buffer.get()

        self.weight_only_flow = torch.empty(
            0,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        self.weight_only_flow.data = freezed_weight

    def get_forward_pre_hook(self):
        def forward_pre_hook(module, input):
            logger.debug(self.debug_msg)
            self.all_gather(self.comm_stream)
            assert self.weight_reusable_buffer is not None
            if self._backward_depth == 0:
                self.weight_reusable_buffer.start_processing(self)
            if self.is_hpz_layer and not self.hpz_inited:
                self.hpz_buffer.copy_(
                    self.weight_reusable_buffer.get().narrow(
                        0, self.hpz_shard_size * self.hpz_rank, self.hpz_shard_size
                    )
                )
                self.hpz_inited = True

        return forward_pre_hook

    def get_forward_hook(self):
        def forward_hook(module, input, output):
            logger.debug(self.debug_msg)
            assert self.weight_reusable_buffer is not None
            if self._backward_depth == 0:
                self.weight_reusable_buffer.end_processing(self)
            # Clear the stale fp8 weight views on the last accumulation step so the
            # next forward cannot hand out views into a buffer that has been freed.
            if self._is_last_accumulation_step and self.is_fp8:
                self._fp8_weights_quantized = None
                self._clear_fp8_quantized_attrs()
            if self.exec_order_tracker:
                self.exec_order_tracker.add_force_module_input(module, output)

        return forward_hook

    def get_backward_pre_hook(self):
        def backward_pre_hook(module, input_grads):
            logger.debug(self.debug_msg)
            self._backward_depth += 1
            assert self.weight_reusable_buffer is not None
            if self.is_trainable:
                grad_buffer_is_same = self.grad_reusable_buffer.is_gained(self)
                self.grad_reusable_buffer.start_prepare(self)
                self.all_gather(self.comm_stream)
                if not grad_buffer_is_same and not self.grad_reusable_buffer.is_super_tensor_superset_of_previous_owner():
                    self.grad_reusable_buffer.get().zero_()
                self.weight_reusable_buffer.start_processing(self)
            else:
                self.all_gather(self.comm_stream)
                self.weight_reusable_buffer.start_processing(self)

        return backward_pre_hook

    def get_backward_hook(self):
        def backward_hook():
            logger.debug(self.debug_msg)
            assert self.weight_reusable_buffer is not None
            self.weight_reusable_buffer.end_processing(self)
            self._backward_depth = max(0, self._backward_depth - 1)
            if self.is_trainable:
                self.grad_reusable_buffer.end_prepare(self)
                if self._zero_stage == 1 and self._is_last_accumulation_step:
                    self.grad_reusable_buffer.free()

        return backward_hook

    def _materialize_params(self, weight_buffer_trainable, weight_buffer_freezed):
        if self.params:
            assert len(weight_buffer_trainable) != 0
            materialize_params(
                self.params,
                weight_buffer_trainable,
                self.grad_reusable_buffer.get(),
            )
        if self.params_freezed:
            assert len(weight_buffer_freezed) != 0
            materialize_params(
                self.params_freezed,
                weight_buffer_freezed,
                None, # don't need grads here => None
            )

    def pre_forward(self, inputs : list[Optional[torch.Tensor]]):
        logger.debug(self.debug_msg)
        is_inputs_trainable = any(x.requires_grad for x in inputs if isinstance(x, torch.Tensor))

        # if inputs / weights / separate modules are trainable -> we need to make backward throw this module
        if is_inputs_trainable or self.must_be_gated:
            logger.debug(f"GatedGF on inputs and weights {self.debug_msg}")
            gated_ = GateGradFlow.apply(self.get_backward_hook(),
                                        self.weight_grad_flow,
                                        *inputs)
            weight_grad_flow = gated_[0]
            output = list(gated_)[1:]
        else:
            # for fully freezed layernorm_supertensor with inputs which may have requires_grad=False
            # (usuall setup for llama, where inputs is input_ids with dtype torch.Long )
            # we skip this GGF and this backward_hook won't be called.
            # So we need to call it manually in GigaFSDP backward hook.
            self._need_manual_backward_hook = True
            weight_grad_flow = self.weight_grad_flow
            output = inputs

        self._materialize_params(weight_grad_flow, self.weight_only_flow)
        return output

    def collect_module(self):
        if self.fp8_active:
            # The gathered buffer holds fp8, so re-gather the bf16 master shard into
            # the scratch purely to materialise readable params (state_dict path).
            bf16_view = self.bf16_weight_scratch.narrow(0, 0, self.padded_numel)
            dist.all_gather_into_tensor(bf16_view, self.shard, group=self._fsdp_process_group)
            self._materialize_params(bf16_view, bf16_view.narrow(0, 0, 0))
        else:
            self.all_gather(self.comm_stream)
            self.weight_reusable_buffer.start_processing(self)
            self._materialize_params(*(self._split_weight_buffer(self.weight_reusable_buffer.get())))
            self.weight_reusable_buffer.end_processing(self)


class SuperTensorModule(torch.nn.Module):
    def __init__(
        self,
        root_module,
        comm_stream,
        name,
        layer_norm_module_cls: Type[nn.Module],
        device: torch.device,
        pg_kit: ProcessGroupKit,
        is_hpz_layer=False,
        hpz_all_gather_stream=None,
        sync_module_states: bool = False,
        param_init_fn: Optional[Callable] = None,
        exec_order_tracker: Optional[ExecutionOrderTracker] = None,
        zero_stage: int = 3,
        alignment_value: Optional[int] = None,
        fp8_all_gather_selected: bool = False,
        param_dtype: torch.dtype = torch.bfloat16,
    ):
        super(SuperTensorModule, self).__init__()
        self.name = name
        self.root_module = root_module
        self.exec_order_tracker = exec_order_tracker
        self.pg_kit = pg_kit

        _init_module(root_module,
                     param_init_fn=param_init_fn,
                     sync_module_states=sync_module_states,
                     fsdp_process_group=pg_kit.fsdp_process_group,
                     device=device
                    )

        if layer_norm_module_cls is None:
            ignored_modules = []
        else:
            ignored_modules = list(filter(lambda m: isinstance(m, layer_norm_module_cls), root_module.modules()))
        is_ignored_module_trainable = any([any(module_params_requires_grad(m)) for m in ignored_modules])

        def get_named_parameters_to_flatten(root_module):
            result = []
            def f(module, module_name):
                if module not in ignored_modules:
                    for param_name, param in module.named_parameters(module_name, recurse=False):
                        result.append((param_name, param))
                    for submodule_name, submodule in module.named_children():
                        submodule_name = f'{module_name}.{submodule_name}' if module_name else submodule_name
                        f(submodule, submodule_name)

            f(root_module, '')
            return result

        named_parameters_to_flatten = get_named_parameters_to_flatten(root_module)

        params_to_flatten = self._init_metaparams(named_parameters_to_flatten, is_trainable=True)
        params_to_flatten_freezed = self._init_metaparams(named_parameters_to_flatten, is_trainable=False)

        self.super_tensor = SuperTensor(
            params_to_flatten,
            params_to_flatten_freezed,
            comm_stream,
            name,
            fsdp_process_group=self.pg_kit.fsdp_process_group,
            intra_node_fsdp_process_group=self.pg_kit.intra_node_fsdp_process_group,
            model_parallel_process_group=self.pg_kit.model_parallel_process_group,
            inter_hybrid_process_group=self.pg_kit.inter_hybrid_process_group,
            is_hpz_layer=is_hpz_layer,
            hpz_all_gather_stream=hpz_all_gather_stream,
            exec_order_tracker=exec_order_tracker,
            is_ignored_module_trainable=is_ignored_module_trainable,
            zero_stage=zero_stage,
            alignment_value=alignment_value,
            fp8_all_gather_selected=fp8_all_gather_selected,
            param_dtype=param_dtype,
        )

        self.register_forward_pre_hook(self.super_tensor.get_forward_pre_hook())
        self.register_forward_hook(self.super_tensor.get_forward_hook())
        self.register_full_backward_pre_hook(self.super_tensor.get_backward_pre_hook())

        if self.exec_order_tracker is not None:
            self.exec_order_tracker.track(self)

    def _init_metaparams(
            self,
            named_parameters : List[Tuple[str, torch.Tensor]],
            is_trainable : bool = True
        ) -> List[MetaParam]:
        filtered_named_parameters = [np for np in named_parameters
                                     if np[1].requires_grad == is_trainable]

        params_to_flatten = [param for _, param in filtered_named_parameters]
        params_to_flatten = convert_some_params_to_metaparams(
            self.root_module, self.name, lambda param: any(param is p for p in params_to_flatten)
        )
        return params_to_flatten

    def pre_forward(self, args, kwargs):
        forced_input = tuple()
        if self.exec_order_tracker is not None:
            forced_input = self.exec_order_tracker.get_force_module_input(self)
        all_inputs = list(args) + [kwargs[key] for key in kwargs]

        forced_input = self.check_forced_input(all_inputs, forced_input)
        all_inputs += list(forced_input)

        gate_result = self.super_tensor.pre_forward(all_inputs)
        args = tuple(gate_result[0 : len(args)])
        for key, gated_input in zip(kwargs, gate_result[len(args) : len(args) + len(kwargs)]):
            kwargs[key] = gated_input
        forced_input = tuple(gate_result[len(args) + len(kwargs):])

        return args, kwargs, forced_input

    def forward(self, *args, **kwargs):
        # forced_input is previous layer output which propagated throw GatedGradFlow of current layer
        # in pre_forward method. But we have to make autograd compute backprop throw forced_input, so we need
        # tie current module computational graph with forced_input without affecting its output.
        # Therefore we apply TyingGate to output to make fake node of computational graph for autograd.
        args, kwargs, forced_input = self.pre_forward(args, kwargs)
        output = self.root_module.forward(*args, **kwargs)
        output = self.tie_forced_input(output, forced_input)
        return output

    def check_forced_input(
        self,
        inputs: list[torch.Tensor],
        forced_input: tuple[Optional[torch.Tensor]]
    ) -> tuple[Optional[torch.Tensor]]:
        assert isinstance(forced_input, tuple), "forced_input must be tuple[Optional[torch.Tensor]]"

        def get_storage_ids(tensors):
            return set(id(t.untyped_storage())
                       for t in tensors
                       if isinstance(t, torch.Tensor)) # for the inputs

        # If previous layer output already have been passed to current layer
        # we needn't to pass it again and we can avoid from adding TyingGate operation
        intersection = get_storage_ids(inputs) & get_storage_ids(forced_input)
        if len(intersection):
            forced_input = tuple()

        return forced_input

    def tie_forced_input(
        self,
        output: Union[torch.Tensor, tuple[torch.Tensor]],
        forced_input: tuple[Optional[torch.Tensor]]
    ) -> Union[torch.Tensor, tuple[torch.Tensor]]:
        # case when forced_input don't have requires_grad=True tensors
        # or have been already passed to input
        if len(forced_input) == 0:
            return output

        logger.debug(f"Add forced input to {self.name} module")
        summed = self.get_forced_input_elem(forced_input)
        output = tree_map_only(torch.Tensor, lambda x: TyingGate.apply(x, summed), output)

        return output

    def get_forced_input_elem(self, forced_input : tuple[Optional[torch.Tensor]]) -> torch.Tensor:
        return sum(0.0 * x.flatten()[0] for x in forced_input)

    def init_layerwise_params(self,
                              main_parameters : Optional[torch.Tensor],
                              freezed_parameters : Optional[torch.Tensor],
                              skip_freezed_params : bool = False,
                              suffix : str = "",
                              ):
        if freezed_parameters is None and not skip_freezed_params:
            freezed_parameters = main_parameters

        self.register_parameter(f"flatten{suffix}",
                                self._setup_param(main_parameters,
                                                  self.super_tensor.shard_offset,
                                                  self.super_tensor.shard_size))

        if freezed_parameters is not None:
            self.register_parameter(f"flatten_freezed{suffix}",
                                    self._setup_param(freezed_parameters,
                                                      self.super_tensor.shard_offset_freezed,
                                                      self.super_tensor.shard_size_freezed))

    def _setup_param(self,
                     weight_buffer : Optional[torch.Tensor],
                     offset : int,
                     shard_size : int) -> Optional[nn.Parameter]:
        if shard_size == 0:
            return

        param = nn.Parameter(
            torch.empty(0,
                dtype=weight_buffer.dtype,
                device=weight_buffer.device,
            ),
            requires_grad=weight_buffer.requires_grad
        )
        param.data = weight_buffer.narrow(0, offset, shard_size)
        if weight_buffer.requires_grad:
            param.grad = weight_buffer.grad.narrow(0, offset, shard_size)
        else:
            param.grad = None
        return param

    def adopt_metaparams(self, main_param):
        flatten_buffer = self.super_tensor.weight_reusable_buffer.get()
        shard = flatten_buffer.narrow(
            0, self.super_tensor.shard_size * self.super_tensor.rank, self.super_tensor.shard_size
        )  # .clone()
        with torch.no_grad():
            main_param[
                self.super_tensor.shard_offset : self.super_tensor.shard_offset + self.super_tensor.shard_size
            ].copy_(shard)

    def __getattr__(self, name: str) -> Union[torch.Tensor, "Module"]:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.root_module, name)
