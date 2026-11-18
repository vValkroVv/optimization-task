import functools
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch._logging import warning_once  # type: ignore[attr-defined]
from torch.autograd import Variable
from torch.autograd.graph import _MultiHandle
from torch.distributed._composable_state import (
    _get_module_state,
    _insert_module_state,
    _State,
)
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.utils import _apply_to_tensors, _to_kwargs
from torch.utils._pytree import tree_flatten

from ._api import MixedPrecisionPolicy
from ._common import DataParallelMeshInfo, TrainingState, _cast_fp_tensor
from ._param_group import (
    GFSDPBufferContext,
    GFSDPCommContext,
    GFSDPParamGroup,
)

if TYPE_CHECKING:
    from ._param import GFSDPParam

    try:
        import yccl
    except ImportError:
        pass

logger = logging.getLogger("gfsdp")


def _alias_buffer_region(
    source: GFSDPBufferContext, sizes: dict[torch.dtype | None, int]
) -> GFSDPBufferContext:
    """Create a bounded alias using storage-root-relative byte offsets."""
    offsets = {}
    for dtype, size in sizes.items():
        if dtype not in source.buffers:
            raise ValueError(f"Buffer region does not contain dtype bucket {dtype}")
        if size < 0 or size > source.buffers[dtype].numel():
            raise ValueError(
                f"Requested {size} bytes from a {source.buffers[dtype].numel()}-byte "
                f"{dtype} buffer region"
            )
        # region() offsets are relative to _storage_ctx, even when the source
        # view is nested. Keep this range inside the source's absolute interval.
        offsets[dtype] = source._ranges[dtype][0]
    return source._storage_ctx.region(sizes, offsets)


class GFSDPStateContext:
    def __init__(self) -> None:
        """Initialize state shared by all GFSDP wrappers under one root."""
        # All GFSDP states in the root state's module tree
        self.all_states: list[GFSDPState] = []
        # Whole parameter groups for which the top-level EMA API allocated
        # buffers. Frozen groups are never registered here: their weights do
        # not change, so the main weights serve as the EMA.
        self._ema_tracked_param_groups: set[GFSDPParamGroup] = set()
        # Iteration's forward root runs the once-per-forward logic; this root
        # may not be the overall root set by lazy initialization in cases where
        # only a submodule runs forward (e.g. encoder-only for eval)
        self.iter_forward_root: GFSDPState | None = None
        # Final callback should only be queued once per backward
        self.post_backward_final_callback_queued: bool = False
        # Whether to finalize backward in this backward's final callback
        self.is_last_backward: bool = True
        # Optional user-provided event recorded after optimizer for the
        # all-gather streams to wait on in the root pre-forward
        self.post_optim_event: torch.Event | None = None


class GFSDPState(_State):
    def __init__(self) -> None:
        super().__init__()
        self._fsdp_param_groups: list[GFSDPParamGroup] = []
        self._is_root: bool | None = None  # root set during lazy init
        self._state_ctx = GFSDPStateContext()
        self._comm_ctx = GFSDPCommContext()
        self._training_state: TrainingState = TrainingState.IDLE
        self._states_to_forward_prefetch: list[GFSDPState] = []
        self._states_to_backward_prefetch: list[GFSDPState] = []
        self._modules_to_run_forward: set[nn.Module] = set()
        # ``False`` when user set reshard_after_forward
        # through ``fully_shard`` or ``set_reshard_after_forward``
        self._auto_reshard_after_forward: bool | None = True
        # Set from ``parallelize`` before the first forward, read at lazy init.
        # ``None`` shard order means fall back to ``named_modules()`` traversal.
        self._shard_order: "list[GFSDPState] | None" = None
        self._own_grad_buffer: bool = False
        self._buffer_region_child: GFSDPState | None = None
        self._buffer_region_transition: (
            "tuple[GFSDPState, GFSDPState, GFSDPState] | None"
        ) = None
        self._allow_no_grad_reduce: bool = False

    # Define a separate init since `__init__` is called in the contract
    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        auto_reshard_after_forward: bool,
    ) -> None:
        for module in modules:
            _insert_module_state(module, self)
        self._modules = modules
        self._device = device
        self._device_handle = _get_device_handle(device.type)
        self._mp_policy = mp_policy
        self._auto_reshard_after_forward = auto_reshard_after_forward
        if len(modules) == 1:
            self._pre_forward_hook_handle = modules[0].register_forward_pre_hook(
                self._pre_forward, prepend=True, with_kwargs=True
            )
            self._post_forward_hook_handle = modules[0].register_forward_hook(
                self._post_forward, prepend=False
            )
        else:
            hook_handle = _register_group_forward_hooks(
                modules,
                self._pre_forward,
                self._post_forward,
                self._modules_to_run_forward,
            )
            self._pre_forward_hook_handle = hook_handle
            self._post_forward_hook_handle = hook_handle

    def _root_pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        self._lazy_init()
        if self._state_ctx.iter_forward_root is not None:
            return args, kwargs
        logger.debug("GFSDP::root_pre_forward")
        self._state_ctx.iter_forward_root = self
        with torch.profiler.record_function("GFSDP::root_pre_forward"):
            # Wait for optimizer before implicitly prefetched all-gathers
            if (event := self._state_ctx.post_optim_event) is not None:
                self._comm_ctx.all_gather_stream.wait_event(event)
                self._state_ctx.post_optim_event = None
            else:
                current_stream = self._device_handle.current_stream()
                self._comm_ctx.all_gather_stream.wait_stream(current_stream)
            if self._device.type == "cuda":
                with torch.profiler.record_function("GFSDP::inputs_to_device"):
                    args_tuple, kwargs_tuple = _to_kwargs(
                        args, kwargs, self._device, False
                    )  # same as DDP
                args, kwargs = args_tuple[0], kwargs_tuple[0]
        return args, kwargs

    def _lazy_init(
        self,
        allow_no_grad_reduce: bool = False,  # noqa: FBT001
        yccl_handle: "yccl.Handle | None" = None,
        process_group_to_yccl_handle: "dict[torch.distributed.ProcessGroup, yccl.Handle] | None" = None,
        param_group_to_data_buffer_ctx_idx: dict[GFSDPParamGroup, int] | None = None,
        param_group_to_grad_buffer_ctx_idx: dict[GFSDPParamGroup, int] | None = None,
        data_buffer_ctx_idx_to_yccl_handle: dict[int, "yccl.Handle"] | None = None,
        grad_buffer_ctx_idx_to_yccl_handle: dict[int, "yccl.Handle"] | None = None,
    ) -> None:
        if self._is_root is not None:
            return  # no-op: already initialized
        # Lazy init runs from the first forward, with no caller left to pass it.
        allow_no_grad_reduce = allow_no_grad_reduce or self._allow_no_grad_reduce
        self._is_root = True
        if len(self._modules) > 1:
            raise RuntimeError(
                f"GFSDP requires a single root module but got {self._modules}"
            )
        root_module = self._modules[0]
        visited_states: set[GFSDPState] = set()
        for module_name, module in root_module.named_modules():
            if (state := _get_module_fsdp_state(module)) is None:
                continue
            if module is not root_module:
                if state not in visited_states and state._is_root is not None:
                    raise RuntimeError(
                        "GFSDP state has already been lazily initialized for "
                        f"{module_name}\nGFSDP requires running forward through "
                        "the root module first"
                    )
                state._is_root = False
            if state not in visited_states:
                self._state_ctx.all_states.append(state)
            visited_states.add(state)
        self._init_fqns()
        self._init_shared_state(
            allow_no_grad_reduce,
            yccl_handle,
            process_group_to_yccl_handle,
            param_group_to_data_buffer_ctx_idx,
            param_group_to_grad_buffer_ctx_idx,
            data_buffer_ctx_idx_to_yccl_handle,
            grad_buffer_ctx_idx_to_yccl_handle,
        )
        self.validate_shared_state()
        # Run parameter group lazy inits after initializing FQNs for improved
        # error messages
        for state in self._state_ctx.all_states:
            for fsdp_param_group in state._fsdp_param_groups:
                fsdp_param_group.lazy_init()
        requested_cache_groups = [
            group
            for state in self._state_ctx.all_states
            for group in state._fsdp_param_groups
            if group._bf16_cache_requested
        ]
        if requested_cache_groups and not any(
            group._bf16_cache_enabled for group in requested_cache_groups
        ):
            warning_once(
                logger,
                "BF16 expert weight reuse was requested, but no GFSDP parameter "
                "group met the supported layout; using the ordinary all-gather path",
                stacklevel=2,
            )

    def _init_shared_state(
        self,
        allow_no_grad_reduce: bool,
        yccl_handle: "yccl.Handle | None",
        process_group_to_yccl_handle: "dict[torch.distributed.ProcessGroup, yccl.Handle] | None",
        param_group_to_data_buffer_ctx_idx: dict[GFSDPParamGroup, int] | None,
        param_group_to_grad_buffer_ctx_idx: dict[GFSDPParamGroup, int] | None,
        data_buffer_ctx_idx_to_yccl_handle: "dict[int, yccl.Handle] | None",
        grad_buffer_ctx_idx_to_yccl_handle: "dict[int, yccl.Handle] | None",
    ) -> None:
        """Share communication and EMA state across the root module tree."""
        self._comm_ctx.lazy_init(self._device)
        for state in self._state_ctx.all_states:
            self._state_ctx._ema_tracked_param_groups.update(
                state._state_ctx._ema_tracked_param_groups
            )
        for state in self._state_ctx.all_states:
            state._state_ctx = self._state_ctx
            state._comm_ctx = self._comm_ctx
            for fsdp_param_group in state._fsdp_param_groups:
                fsdp_param_group.comm_ctx = self._comm_ctx

        ordered_states = _forward_ordered_states(
            self._state_ctx.all_states, self._shard_order
        )

        param_groups = [
            fsdp_param_group
            for state in ordered_states
            for fsdp_param_group in state._fsdp_param_groups
        ]

        region_children = {}
        for state in ordered_states:
            if (child := state._buffer_region_child) is None:
                continue
            if len(state._fsdp_param_groups) != 1 or len(child._fsdp_param_groups) != 1:
                raise ValueError(
                    "Shared buffer regions require one parameter group per unit"
                )
            parent_group = state._fsdp_param_groups[0]
            child_group = child._fsdp_param_groups[0]
            if (
                parent_group.mesh_info is not child_group.mesh_info
                and parent_group.mesh_info != child_group.mesh_info
            ) or (
                parent_group.mp_policy != child_group.mp_policy
                or parent_group.orig_dtype != child_group.orig_dtype
                or parent_group._shard_alignment != child_group._shard_alignment
                or parent_group.requires_grad != child_group.requires_grad
            ):
                raise ValueError(
                    "Shared buffer regions require matching mesh and precision"
                )
            region_children[parent_group] = child_group
        if region_children and (
            yccl_handle is not None
            or process_group_to_yccl_handle
            or param_group_to_data_buffer_ctx_idx is not None
            or param_group_to_grad_buffer_ctx_idx is not None
            or data_buffer_ctx_idx_to_yccl_handle is not None
            or grad_buffer_ctx_idx_to_yccl_handle is not None
        ):
            raise ValueError(
                "Shared buffer regions do not support custom buffer maps or YCCL"
            )
        child_groups = set(region_children.values())
        # Deal the original units, so splitting cannot change alternating slots.
        param_groups = [group for group in param_groups if group not in child_groups]

        def buffer_sizes(group: GFSDPParamGroup, *, grad: bool) -> dict:
            itemsize = (
                group.reduce_dtype or group.param_dtype or group.orig_dtype
            ).itemsize
            return {
                dtype: numel
                * (
                    itemsize
                    if grad
                    else (dtype or group.param_dtype or group.orig_dtype).itemsize
                )
                for dtype, numel in group.padded_unsharded_param_numel.items()
            }

        def original_buffer_sizes(group: GFSDPParamGroup, *, grad: bool) -> dict:
            if group not in region_children:
                return buffer_sizes(group, grad=grad)
            child = region_children[group]
            divider = group.mesh_info.shard_mesh_size * group._shard_alignment
            sizes = {}
            for dtype in (
                group.padded_unsharded_param_numel.keys()
                | child.padded_unsharded_param_numel.keys()
            ):
                numel = sum(group._unsharded_param_numels.get(dtype, ())) + sum(
                    child._unsharded_param_numels.get(dtype, ())
                )
                padded = (numel + divider - 1) // divider * divider
                sizes[dtype] = (
                    padded
                    * (
                        (group.reduce_dtype or group.param_dtype or group.orig_dtype)
                        if grad
                        else (dtype or group.param_dtype or group.orig_dtype)
                    ).itemsize
                )
            return sizes

        def partition(
            group: GFSDPParamGroup, ctx: GFSDPBufferContext, *, grad: bool
        ) -> None:
            if group not in region_children:
                return
            child = region_children[group]
            parent_sizes, child_sizes = (
                buffer_sizes(group, grad=grad),
                buffer_sizes(child, grad=grad),
            )
            for dtype in parent_sizes.keys() | child_sizes.keys():
                if (
                    parent_sizes.get(dtype, 0) + child_sizes.get(dtype, 0)
                    > ctx.buffers[dtype].numel()
                ):
                    raise ValueError(
                        "Split buffer regions exceed the original allocation budget"
                    )
            attr = "grad_buffer_ctx" if grad else "data_buffer_ctx"
            setattr(
                group, attr, ctx.region(parent_sizes, dict.fromkeys(parent_sizes, 0))
            )
            setattr(
                child,
                attr,
                ctx.region(
                    child_sizes,
                    {dtype: parent_sizes.get(dtype, 0) for dtype in child_sizes},
                ),
            )

        no_reshard_after_forward = all(
            not param_group.reshard_after_forward for param_group in param_groups
        )
        data_buffer_ctx2ctx_using_param_groups, data_buffer_ctx2yccl_handle = (
            get_buffer_ctx2ctx_using_param_groups_map(
                param_groups,
                "data_buffer_ctx",
                None
                if no_reshard_after_forward or param_group_to_data_buffer_ctx_idx
                else 2,
                yccl_handle,
                process_group_to_yccl_handle,
                param_group_to_data_buffer_ctx_idx,
                data_buffer_ctx_idx_to_yccl_handle,
            )
        )
        grad_param_groups = [
            param_group for param_group in param_groups if param_group.requires_grad
        ]

        # One buffer of its own for the unit that asked; the rest share, as
        # before. Pointless under ``allow_no_grad_reduce``: everyone has one.
        own_grad_buffer = {
            param_group
            for state in ordered_states
            if state._own_grad_buffer
            for param_group in state._fsdp_param_groups
        }
        if (
            param_group_to_grad_buffer_ctx_idx is None
            and not allow_no_grad_reduce
            and not own_grad_buffer.isdisjoint(grad_param_groups)
        ):
            param_group_to_grad_buffer_ctx_idx = _grad_buffer_ctx_idx_map(
                grad_param_groups, own_grad_buffer
            )

        grad_buffer_ctx2ctx_using_param_groups, grad_buffer_ctx2yccl_handle = (
            get_buffer_ctx2ctx_using_param_groups_map(
                grad_param_groups,
                "grad_buffer_ctx",
                # An explicit map leaves no count to alternate by.
                None
                if allow_no_grad_reduce or param_group_to_grad_buffer_ctx_idx
                else 1,
                yccl_handle,
                process_group_to_yccl_handle,
                param_group_to_grad_buffer_ctx_idx,
                grad_buffer_ctx_idx_to_yccl_handle,
            )
        )

        for (
            data_buffer_ctx,
            ctx_using_param_groups,
        ) in data_buffer_ctx2ctx_using_param_groups.items():
            all_gather_dtypes = {
                dtype
                for param_group in ctx_using_param_groups
                for dtype in original_buffer_sizes(param_group, grad=False)
            }
            default_handle = data_buffer_ctx2yccl_handle[data_buffer_ctx]
            data_buffer_ctx.lazy_init(
                {
                    dtype: max(
                        original_buffer_sizes(param_group, grad=False).get(dtype, 0)
                        for param_group in ctx_using_param_groups
                    )
                    for dtype in all_gather_dtypes
                },
                self._device,
                yccl_handles=dict.fromkeys(all_gather_dtypes, default_handle),
            )
            for param_group in ctx_using_param_groups:
                param_group.data_buffer_ctx = data_buffer_ctx
                partition(param_group, data_buffer_ctx, grad=False)
            if self._auto_reshard_after_forward:
                # Do not reshard the last module across modules using the same data buffer
                # after forward since for training the parameters would be
                # freed and all-gathered immediately
                ctx_using_param_groups[-1].post_forward_mesh_info = None
                if (
                    child := region_children.get(ctx_using_param_groups[-1])
                ) is not None:
                    child.post_forward_mesh_info = None
        for (
            grad_buffer_ctx,
            ctx_using_param_groups,
        ) in grad_buffer_ctx2ctx_using_param_groups.items():
            all_gather_dtypes = {
                dtype
                for param_group in ctx_using_param_groups
                for dtype in original_buffer_sizes(param_group, grad=True)
            }
            default_handle = grad_buffer_ctx2yccl_handle[grad_buffer_ctx]
            grad_buffer_ctx.lazy_init(
                {
                    dtype: max(
                        original_buffer_sizes(param_group, grad=True).get(dtype, 0)
                        for param_group in ctx_using_param_groups
                    )
                    for dtype in all_gather_dtypes
                },
                self._device,
                yccl_handles=dict.fromkeys(all_gather_dtypes, default_handle),
            )
            for param_group in ctx_using_param_groups:
                param_group.grad_buffer_ctx = grad_buffer_ctx
                partition(param_group, grad_buffer_ctx, grad=True)

        if self._buffer_region_transition is not None:
            source_state, data_target_state, grad_target_state = (
                self._buffer_region_transition
            )
            if any(
                state not in ordered_states
                for state in (source_state, data_target_state, grad_target_state)
            ):
                raise ValueError(
                    "Buffer region transition modules must belong to the root state"
                )
            if len(source_state._fsdp_param_groups) != 1:
                raise ValueError(
                    "Buffer region transition source must have exactly one "
                    "parameter group"
                )
            source_group = source_state._fsdp_param_groups[0]
            source_process_group = source_group.mesh_info.shard_process_group

            def matching_group(
                state: GFSDPState, name: str
            ) -> GFSDPParamGroup:
                matches = [
                    group
                    for group in state._fsdp_param_groups
                    if group.mesh_info.shard_process_group is source_process_group
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Buffer region transition requires one {name} parameter "
                        f"group on the source dense process group; got {len(matches)}"
                    )
                return matches[0]

            data_target_group = matching_group(data_target_state, "data target")
            grad_target_group = matching_group(grad_target_state, "gradient target")

            def alias_target(
                source_ctx: GFSDPBufferContext,
                target_group: GFSDPParamGroup,
                *,
                grad: bool,
            ) -> tuple[GFSDPBufferContext, dict[torch.dtype | None, int]]:
                if (
                    source_group.mesh_info.shard_mesh_size
                    != target_group.mesh_info.shard_mesh_size
                    or source_group.mesh_info.shard_mesh_rank
                    != target_group.mesh_info.shard_mesh_rank
                    or source_group._shard_alignment != target_group._shard_alignment
                    or source_group.param_dtype != target_group.param_dtype
                    or source_group.reduce_dtype != target_group.reduce_dtype
                    or source_group.orig_dtype != target_group.orig_dtype
                    or source_group.mp_policy != target_group.mp_policy
                ):
                    raise ValueError(
                        "Buffer region transition groups must have matching mesh "
                        "and precision"
                    )
                sizes = buffer_sizes(target_group, grad=grad)
                offsets = {}
                for dtype, size in sizes.items():
                    if dtype not in source_ctx.buffers:
                        raise ValueError(
                            "Buffer region transition dtype buckets do not match: "
                            f"{dtype} is missing from the MTP FFN region"
                        )
                    start, end = source_ctx._ranges[dtype]
                    item_dtype = (
                        target_group.reduce_dtype
                        or target_group.param_dtype
                        or target_group.orig_dtype
                        if grad
                        else dtype or target_group.param_dtype or target_group.orig_dtype
                    )
                    alignment = (
                        target_group.mesh_info.shard_mesh_size
                        * target_group._shard_alignment
                        * item_dtype.itemsize
                    )
                    capacity = end - start
                    if size > capacity:
                        raise ValueError(
                            "Buffer region transition exceeds MTP FFN region "
                            f"capacity for {dtype}: need {size} bytes, have {capacity}"
                        )
                    if start % alignment or size % alignment:
                        raise ValueError(
                            "Buffer region transition is not shard-aligned for "
                            f"{dtype}: offset={start}, size={size}, alignment={alignment}"
                        )
                    offsets[dtype] = start
                return _alias_buffer_region(source_ctx, sizes), offsets

            source_data_ctx = source_group.data_buffer_ctx
            source_grad_ctx = source_group.grad_buffer_ctx
            if source_grad_ctx is None:
                raise ValueError(
                    "Buffer region transition source has no gradient buffer"
                )
            data_alias, data_offsets = alias_target(
                source_data_ctx, data_target_group, grad=False
            )
            grad_alias, grad_offsets = alias_target(
                source_grad_ctx, grad_target_group, grad=True
            )
            data_target_group.data_buffer_ctx = data_alias
            grad_target_group.grad_buffer_ctx = grad_alias
            source_group._post_backward_prefetch_target = data_target_group

            def alias_summary(
                alias: GFSDPBufferContext,
                source_ctx: GFSDPBufferContext,
                offsets: dict[torch.dtype | None, int],
            ) -> dict[str, dict[str, int]]:
                return {
                    str(dtype): {
                        "offset_bytes": offsets[dtype],
                        "size_bytes": size,
                        "capacity_bytes": source_ctx.buffers[dtype].numel(),
                        "pool_bytes": alias._storage_ctx.buffers[dtype].numel(),
                    }
                    for dtype, size in (
                        (dtype, tensor.numel()) for dtype, tensor in alias.buffers.items()
                    )
                }

            logger.info(
                "GFSDP buffer region transition active: source=%s data_target=%s "
                "grad_target=%s data=%s grad=%s",
                source_group._module_fqn,
                data_target_group._module_fqn,
                grad_target_group._module_fqn,
                alias_summary(data_alias, source_data_ctx, data_offsets),
                alias_summary(grad_alias, source_grad_ctx, grad_offsets),
            )

    def validate_shared_state(self) -> None:
        data_buffer_ctx2ctx_using_param_groups: dict[
            GFSDPBufferContext, list[GFSDPParamGroup]
        ] = {}
        grad_buffer_ctx2ctx_using_param_groups: dict[
            GFSDPBufferContext, list[GFSDPParamGroup]
        ] = {}
        for state in self._state_ctx.all_states:
            for param_group in state._fsdp_param_groups:
                data_buffer_ctx2ctx_using_param_groups.setdefault(
                    param_group.data_buffer_ctx, []
                ).append(param_group)
                if param_group.grad_buffer_ctx is not None:
                    grad_buffer_ctx2ctx_using_param_groups.setdefault(
                        param_group.grad_buffer_ctx, []
                    ).append(param_group)
        if (num_data_buffers := len(data_buffer_ctx2ctx_using_param_groups)) < min(
            2,
            num_param_groups := sum(
                map(len, data_buffer_ctx2ctx_using_param_groups.values())
            ),
        ):
            raise ValueError(
                "num_data_buffers must be no less than min(2, num_param_groups)"
                f" for correct overlap, but got {num_data_buffers=} {num_param_groups=}."
            )
        if (num_grad_buffers := len(grad_buffer_ctx2ctx_using_param_groups)) < min(
            1,
            num_param_groups := sum(
                map(len, grad_buffer_ctx2ctx_using_param_groups.values())
            ),
        ):
            raise ValueError(
                "num_grad_buffers must be no less than min(1, num_param_groups)"
                f", but got {num_grad_buffers=} {num_param_groups=}."
            )
        data_param_groups = [
            param_group
            for groups in data_buffer_ctx2ctx_using_param_groups.values()
            for param_group in groups
        ]
        for index, param_group in enumerate(data_param_groups):
            for other in data_param_groups[index + 1 :]:
                if not param_group.data_buffer_ctx.overlaps(other.data_buffer_ctx):
                    continue
                if not param_group.reshard_after_forward and not other.reshard_after_forward:
                    raise ValueError(
                        "Only one parameter group across groups which share a data buffer"
                        " is allowed to have no reshard after forward, but got "
                        f"{param_group=} and {other=}"
                    )
                if not param_group.reshard_after_backward and not other.reshard_after_backward:
                    raise ValueError(
                        "Only one parameter group across groups which share a data buffer"
                        " is allowed to have no reshard after backward, but got "
                        f"{param_group=} and {other=}"
                    )

        grad_param_groups = [
            param_group
            for groups in grad_buffer_ctx2ctx_using_param_groups.values()
            for param_group in groups
        ]
        for index, param_group in enumerate(grad_param_groups):
            grad_buffer_ctx = param_group.grad_buffer_ctx
            assert grad_buffer_ctx is not None
            for other in grad_param_groups[index + 1 :]:
                other_grad_buffer_ctx = other.grad_buffer_ctx
                assert other_grad_buffer_ctx is not None
                if (
                    grad_buffer_ctx.overlaps(other_grad_buffer_ctx)
                    and not param_group.reduce_grads
                    and not other.reduce_grads
                ):
                    raise ValueError(
                        "Only one parameter group across groups which share a grad buffer"
                        " is allowed to have no grad reduce, but got "
                        f"{param_group=} and {other=}"
                    )

        for state in self._state_ctx.all_states:
            for param_group in state._fsdp_param_groups:
                if param_group._post_backward_prefetch_target is None:
                    continue
                target = param_group._post_backward_prefetch_target
                if not param_group.reshard_after_backward:
                    raise ValueError(
                        "MTP transition source must reshard after backward to "
                        "release its data region"
                    )
                if not target.reshard_after_forward:
                    raise ValueError(
                        "MTP transition data target must reshard after forward "
                        "to release the shared region"
                    )
                if not target.reshard_after_backward:
                    raise ValueError(
                        "MTP transition data target must reshard after backward "
                        "to release the region for the next microbatch"
                    )
        data_buffer_ctx2module_prefix: dict[GFSDPBufferContext, str] = {}
        grad_buffer_ctx2module_prefix: dict[GFSDPBufferContext, str] = {}
        state2modules_to_run_forward: dict[GFSDPState, set[nn.Module]] = {}

        def validate_buffers(module: nn.Module, prefix: str = "") -> None:
            if (
                state := _get_module_fsdp_state(module)
            ) is not None and state._fsdp_param_groups:
                if state not in state2modules_to_run_forward:
                    state2modules_to_run_forward[state] = set(state._modules)
                    for param_group in state._fsdp_param_groups:
                        data_buffer_ctx = param_group.data_buffer_ctx
                        conflicting = next(
                            (
                                ctx
                                for ctx in data_buffer_ctx2module_prefix
                                if data_buffer_ctx.overlaps(ctx)
                            ),
                            None,
                        )
                        if conflicting is not None:
                            raise RuntimeError(
                                f"Param groups of {prefix} and {data_buffer_ctx2module_prefix[conflicting]} share a"
                                " data buffer, so they can't be unsharded at the same time."
                            )
                        data_buffer_ctx2module_prefix[data_buffer_ctx] = prefix
                        if (grad_buffer_ctx := param_group.grad_buffer_ctx) is not None:
                            conflicting = next(
                                (
                                    ctx
                                    for ctx in grad_buffer_ctx2module_prefix
                                    if grad_buffer_ctx.overlaps(ctx)
                                ),
                                None,
                            )
                            if conflicting is not None:
                                raise RuntimeError(
                                    f"Param groups of {prefix} and {grad_buffer_ctx2module_prefix[conflicting]} share a"
                                    " grad buffer, so they can't be unsharded at the same time."
                                )
                            grad_buffer_ctx2module_prefix[grad_buffer_ctx] = prefix
            for submodule_name, submodule in module.named_children():
                validate_buffers(
                    submodule, prefix + ("." if prefix else "") + submodule_name
                )
            if (
                state := _get_module_fsdp_state(module)
            ) is not None and state._fsdp_param_groups:
                modules_to_run_forward = state2modules_to_run_forward[state]
                modules_to_run_forward.remove(module)
                if len(modules_to_run_forward) == 0:
                    for param_group in state._fsdp_param_groups:
                        del data_buffer_ctx2module_prefix[param_group.data_buffer_ctx]
                        if (grad_buffer_ctx := param_group.grad_buffer_ctx) is not None:
                            del grad_buffer_ctx2module_prefix[grad_buffer_ctx]

        root_module = self._modules[0]
        validate_buffers(root_module)

    def _init_fqns(self) -> None:
        if not self._is_root:
            raise AssertionError("Expected _is_root to be True")
        root_module = self._modules[0]
        param_to_fsdp_param: dict[nn.Parameter, GFSDPParam] = {}
        module_to_fsdp_param_groups: dict[nn.Module, list[GFSDPParamGroup]] = {}
        # A state with sibling groups would otherwise give them one FQN, and the
        # profiler labels and log lines could not be told apart.
        param_group_to_suffix: dict[GFSDPParamGroup, str] = {}
        for state in self._state_ctx.all_states:
            label_shard_axis = len(state._fsdp_param_groups) > 1
            for fsdp_param_group in state._fsdp_param_groups:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    param_to_fsdp_param[fsdp_param.sharded_param] = fsdp_param
                for module in fsdp_param_group.modules:
                    module_to_fsdp_param_groups.setdefault(module, []).append(
                        fsdp_param_group
                    )
                if label_shard_axis:
                    param_group_to_suffix[fsdp_param_group] = (
                        f"[{_shard_axis_name(fsdp_param_group.mesh_info)}]"
                    )
        for param_name, param in root_module.named_parameters():
            if param in param_to_fsdp_param:
                param_to_fsdp_param[param]._param_fqn = param_name
        for module_name, module in root_module.named_modules():
            for fsdp_param_group in module_to_fsdp_param_groups.get(module, ()):
                group_name = module_name + param_group_to_suffix.get(
                    fsdp_param_group, ""
                )
                module_fqn = fsdp_param_group._module_fqn
                if module_fqn is None:
                    fsdp_param_group._module_fqn = group_name
                else:
                    if not isinstance(module_fqn, str):
                        raise AssertionError(
                            f"Expected module_fqn to be str, got {type(module_fqn)}: {module_fqn}"
                        )
                    module_fqn += f", {group_name}"
                    fsdp_param_group._module_fqn = module_fqn

    @torch.compiler.disable  # type: ignore[misc]
    def _pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # When composing with module-hook-based activation checkpointing, the
        # the pre-backward hook is responsible for the unshard
        if self._training_state == TrainingState.PRE_BACKWARD:
            return args, kwargs
        self._training_state = TrainingState.FORWARD
        args, kwargs = self._root_pre_forward(module, args, kwargs)
        if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
            with torch.profiler.record_function("GFSDP::cast_forward_inputs"):
                cast_fn = functools.partial(
                    _cast_fp_tensor, self._mp_policy.param_dtype
                )
                args, kwargs = (
                    _apply_to_tensors(cast_fn, args),
                    _apply_to_tensors(cast_fn, kwargs),
                )
        for fsdp_param_group in self._fsdp_param_groups:
            args, kwargs = fsdp_param_group.pre_forward(module, args, kwargs)
        for fsdp_state in self._states_to_forward_prefetch:
            for target_param_group in fsdp_state._fsdp_param_groups:
                GFSDPParamGroup._prefetch_unshard(target_param_group, "forward")
        return args, kwargs

    @torch.compiler.disable  # type: ignore[misc]
    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        # When composing with module-hook-based activation checkpointing, the
        # post-backward hook is responsible for the reshard
        if self._training_state == TrainingState.PRE_BACKWARD:
            return output
        for fsdp_param_group in self._fsdp_param_groups:
            output = fsdp_param_group.post_forward(module, input, output)
        output = self._register_pre_backward_hook(output)
        self._training_state = TrainingState.IDLE
        if self._state_ctx.iter_forward_root is self:
            self._state_ctx.iter_forward_root = None
        if self._mp_policy.output_dtype is not None:
            with torch.profiler.record_function("GFSDP::cast_forward_outputs"):
                output = _apply_to_tensors(
                    functools.partial(_cast_fp_tensor, self._mp_policy.output_dtype),
                    output,
                )
        return output

    def _pre_backward(self, grad: torch.Tensor) -> torch.Tensor:
        self._training_state = TrainingState.PRE_BACKWARD
        self._register_root_post_backward_final_callback()
        default_prefetch = len(self._states_to_backward_prefetch) == 0
        for fsdp_param_group in self._fsdp_param_groups:
            fsdp_param_group.pre_backward(default_prefetch)
        for fsdp_state in self._states_to_backward_prefetch:
            for target_param_group in fsdp_state._fsdp_param_groups:
                GFSDPParamGroup._prefetch_unshard(target_param_group, "backward")
        return grad

    def _root_post_backward_final_callback(self) -> None:
        logger.debug("GFSDP::root_post_backward")
        with torch.profiler.record_function("GFSDP::root_post_backward_callback"):
            for state in self._state_ctx.all_states:
                for fsdp_param_group in state._fsdp_param_groups:
                    if fsdp_param_group._training_state != TrainingState.POST_BACKWARD:
                        # Run post-backward in case forward inputs did not require
                        # gradient so the autograd backward did not run
                        fsdp_param_group.post_backward()
                    fsdp_param_group._training_state = TrainingState.IDLE
                state._training_state = TrainingState.IDLE
                if self._state_ctx.is_last_backward:
                    state._finalize_backward()
                for fsdp_param_group in state._fsdp_param_groups:
                    for (
                        yccl_handle
                    ) in fsdp_param_group.data_buffer_ctx.yccl_handles.values():
                        if yccl_handle is not None:
                            yccl_handle.process_profiling_events()
                    if (
                        grad_buffer_ctx := fsdp_param_group.grad_buffer_ctx
                    ) is not None:
                        for yccl_handle in grad_buffer_ctx.yccl_handles.values():
                            if yccl_handle is not None:
                                yccl_handle.process_profiling_events()
            if self._state_ctx.is_last_backward:
                self._comm_ctx.post_forward_order.clear()
            self._state_ctx.post_backward_final_callback_queued = False

    def _finalize_backward(self) -> None:
        if self._modules_to_run_forward:
            msg = (
                f"{len(self._modules_to_run_forward)} of the {len(self._modules)} "
                f"modules passed to fully_shard did not run forward before backward, "
                "which is error-prone since FSDP post-forward/pre-backward logic "
                "will not run for these modules. We recommend passing only modules "
                "that run forward together. Modules that did not run forward: "
                f"{list(self._modules_to_run_forward)}"
            )
            warning_once(logger, msg, stacklevel=2)
            # Clear since we want the next forward to run
            self._modules_to_run_forward.clear()
        for fsdp_param_group in self._fsdp_param_groups:
            fsdp_param_group.finalize_backward()

    def _register_pre_backward_hook(self, output: Any) -> Any:
        if not torch.is_grad_enabled():
            return output
        flat_outputs, _ = tree_flatten(output)
        for t in flat_outputs:
            if torch.is_tensor(t) and t.requires_grad:
                t.register_hook(self._pre_backward)
        return output

    def _register_root_post_backward_final_callback(self) -> None:
        if self._state_ctx.post_backward_final_callback_queued:
            return
        self._state_ctx.post_backward_final_callback_queued = True
        Variable._execution_engine.queue_callback(
            self._root_post_backward_final_callback
        )


def _shard_axis_name(mesh_info: DataParallelMeshInfo) -> str:
    """Name the mesh axis a group shards over, for FQNs and profiler labels."""
    mesh_axis_names = mesh_info.mesh.mesh_dim_names
    shard_axis = mesh_info.shard_mesh_dim
    if mesh_axis_names is None or shard_axis is None:
        return str(shard_axis)
    return mesh_axis_names[shard_axis]


def _get_module_fsdp_state(module: nn.Module) -> GFSDPState | None:
    state = _get_module_state(module)
    if isinstance(state, GFSDPState):
        return state
    return None


def _register_group_forward_hooks(
    modules: Sequence[nn.Module],
    pre_hook: Callable,  # type: ignore[type-arg]
    post_hook: Callable,  # type: ignore[type-arg]
    modules_to_run: set[nn.Module],
) -> _MultiHandle:
    modules_set = set(modules)

    @functools.wraps(pre_hook)
    @torch.compiler.disable  # type: ignore[misc]
    def wrapped_pre_hook(*args: Any, **kwargs: Any) -> Any | None:
        if len(modules_to_run) == 0:  # first to run
            modules_to_run.update(modules_set)
            return pre_hook(*args, **kwargs)
        return None

    def get_wrapped_post_hook(module: nn.Module) -> Callable[..., Any | None]:
        @functools.wraps(post_hook)
        @torch.compiler.disable  # type: ignore[misc]
        def wrapped_post_hook(*args: Any, **kwargs: Any) -> Any | None:
            modules_to_run.discard(module)
            if len(modules_to_run) == 0:
                return post_hook(*args, **kwargs)
            return None

        return wrapped_post_hook

    pre_handles = [
        module.register_forward_pre_hook(
            wrapped_pre_hook, prepend=True, with_kwargs=True
        )
        for module in modules
    ]
    post_handles = [
        module.register_forward_hook(
            get_wrapped_post_hook(module), prepend=False, always_call=True
        )
        for module in modules
    ]
    return _MultiHandle(tuple(pre_handles + post_handles))


def _forward_ordered_states(
    all_states: "list[GFSDPState]", shard_order: "list[GFSDPState] | None"
) -> "list[GFSDPState]":
    """Forward order, which is what buffers are dealt along.

    ``all_states`` is declaration order and need not match. Undeclared states
    keep their position after the declared ones; unknown names are ignored.
    """
    if shard_order is None:
        return list(all_states)
    known = set(all_states)
    declared = [state for state in shard_order if state in known]
    seen = set(declared)
    return declared + [state for state in all_states if state not in seen]


def _grad_buffer_ctx_idx_map(
    grad_param_groups: list[GFSDPParamGroup],
    own_grad_buffer: "set[GFSDPParamGroup]",
) -> dict[GFSDPParamGroup, int]:
    """Which gradient buffer each group reduces through.

    A group that asked gets its own index; the rest share one per process group.
    Indices are positions in ``grad_param_groups``, as the caller resolves them.
    """
    shared_idx: dict[torch.distributed.ProcessGroup, int] = {}
    mapping: dict[GFSDPParamGroup, int] = {}
    for index, param_group in enumerate(grad_param_groups):
        if param_group in own_grad_buffer:
            mapping[param_group] = index
            continue
        mapping[param_group] = shared_idx.setdefault(
            param_group.mesh_info.shard_process_group, index
        )
    return mapping


def get_buffer_ctx2ctx_using_param_groups_map(
    param_groups: list[GFSDPParamGroup],
    buffer_ctx_attr_name: str,
    num_buffers_per_process_group: int | None,
    yccl_handle: "yccl.Handle | None",
    process_group_to_yccl_handle: "dict[torch.distributed.ProcessGroup, yccl.Handle] | None",
    param_group_to_buffer_ctx_idx: dict[GFSDPParamGroup, int] | None = None,
    buffer_ctx_idx_to_yccl_handle: "dict[int, yccl.Handle] | None" = None,
) -> tuple[
    dict[GFSDPBufferContext, list[GFSDPParamGroup]],
    "dict[GFSDPBufferContext, list[yccl.Handle | None]]",
]:
    process_group_to_param_groups: dict[
        torch.distributed.ProcessGroup, list[GFSDPParamGroup]
    ] = {}
    for param_group in param_groups:
        process_group_to_param_groups.setdefault(
            param_group.mesh_info.shard_process_group, []
        ).append(param_group)
    if param_group_to_buffer_ctx_idx is None:
        param_group_to_buffer_ctx_idx = {}
        for intra_pg_param_groups in process_group_to_param_groups.values():
            param_group_to_buffer_ctx_idx.update(
                {
                    param_group: (
                        max(param_group_to_buffer_ctx_idx.values(), default=-1)
                        + 1
                        + (
                            intra_pg_index
                            % (
                                num_buffers_per_process_group
                                if num_buffers_per_process_group is not None
                                else len(intra_pg_param_groups)
                            )
                        )
                    )
                    for intra_pg_index, param_group in enumerate(intra_pg_param_groups)
                }
            )
    else:
        assert num_buffers_per_process_group is None
    buffer_ctx2ctx_using_param_groups: dict[
        GFSDPBufferContext, list[GFSDPParamGroup]
    ] = {}
    buffer_ctx2yccl_handle = {}
    for param_group in param_groups:
        buffer_ctx_idx = param_group_to_buffer_ctx_idx[param_group]
        buffer_ctx2ctx_using_param_groups.setdefault(
            (buffer_ctx := getattr(param_groups[buffer_ctx_idx], buffer_ctx_attr_name)),
            [],
        ).append(param_group)
        if buffer_ctx_idx_to_yccl_handle is not None:
            buffer_ctx2yccl_handle[buffer_ctx] = buffer_ctx_idx_to_yccl_handle[
                buffer_ctx_idx
            ]
        elif process_group_to_yccl_handle is not None:
            buffer_ctx2yccl_handle[buffer_ctx] = process_group_to_yccl_handle[
                param_group.mesh_info.shard_process_group
            ]
        else:
            buffer_ctx2yccl_handle[buffer_ctx] = yccl_handle
    return buffer_ctx2ctx_using_param_groups, buffer_ctx2yccl_handle
