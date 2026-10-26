from collections.abc import Iterable, Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
    cast,
    overload,
)

import torch
import torch.nn as nn
from torch.distributed._composable import contract  # type: ignore[attr-defined]
from torch.distributed._composable.contract import STATE_KEY as _CONTRACT_STATE_KEY
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import fully_shard as _torch_fully_shard
from torch.distributed.utils import _get_root_modules

from ._api import (
    MixedPrecisionPolicy,
)
from ._common import FSDPMeshInfo, HSDPMeshInfo
from ._init import (
    _get_device_from_mesh,
    _get_managed_modules,
    _get_managed_states,
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
    _move_states_to_device,
)
from ._param_group import GFSDPParamGroup
from ._state import GFSDPState, _get_module_fsdp_state
from ._tensor import install as _install_ragged_shard_dtensor

if TYPE_CHECKING:
    from ._collectives import AllGatherResult

__all__ = [
    "UnshardHandle",
    "GFSDPModule",
    "fully_shard",
]

cls_to_fsdp_cls: dict[type, type] = {}


@overload
def fully_shard(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = ...,
    reshard_after_forward: bool | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
    param_meshes: Sequence[tuple[DeviceMesh, set[nn.Parameter]]] | None = ...,
    shard_alignment: int = ...,
) -> "GFSDPModule": ...


@overload
def fully_shard(
    module: list[nn.Module],
    *,
    mesh: DeviceMesh | None = ...,
    reshard_after_forward: bool | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
    param_meshes: Sequence[tuple[DeviceMesh, set[nn.Parameter]]] | None = ...,
    shard_alignment: int = ...,
) -> list["GFSDPModule"]: ...


@contract(state_cls=GFSDPState)  # type: ignore[misc]
def fully_shard(
    module: nn.Module | list[nn.Module],
    *,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | None = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),  # noqa: B008
    ignored_params: set[nn.Parameter] | None = None,
    param_meshes: Sequence[tuple[DeviceMesh, set[nn.Parameter]]] | None = None,
    shard_alignment: int = 8,
) -> "GFSDPModule | list[GFSDPModule]":
    """Shard ``module``'s parameters, optionally over more than one mesh.

    ``param_meshes`` names parameters that shard over a mesh other than
    ``mesh``. Each entry becomes its own parameter group under this one unit,
    so the hooks stay on ``module`` itself.
    """
    if isinstance(module, nn.ModuleList | nn.ModuleDict):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    _install_ragged_shard_dtensor()
    mesh = mesh or _init_default_fully_shard_mesh()
    mesh_info = _get_mesh_info(mesh)
    device = _get_device_from_mesh(mesh)
    # TODO: during validation the last layer's buffer is not released, and the
    # model simply fails on the next validation step. The forward-backward
    # overlap is disabled for now; this needs fixing.
    auto_reshard_after_forward = reshard_after_forward is None
    reshard_after_forward_value = (
        True if auto_reshard_after_forward else reshard_after_forward
    )

    arg_module = module
    modules = (
        (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
    )
    state = fully_shard.state(modules[0])  # type: ignore[attr-defined]
    state.init(modules, device, mp_policy, auto_reshard_after_forward)

    managed_modules = _get_managed_modules(modules, ignored_params)
    params, buffers = _get_managed_states(managed_modules, ignored_params)

    _move_states_to_device(params, buffers, device)
    state._fsdp_param_groups = []
    for group_mesh, group_params in _partition_params(params, param_meshes):
        if group_mesh is None:
            group_mesh_info = mesh_info
        else:
            if (group_device := _get_device_from_mesh(group_mesh)) != device:
                raise ValueError(
                    f"param_meshes entry for {group_mesh} resolves to {group_device},"
                    f" but the unit is on {device}."
                )
            group_mesh_info = _get_mesh_info(group_mesh)
        state._fsdp_param_groups.append(
            GFSDPParamGroup(
                group_params,
                modules,
                group_mesh_info,
                _get_post_forward_mesh_info(
                    reshard_after_forward_value, group_mesh_info
                ),
                device,
                mp_policy,
                shard_alignment,
            )
        )

    # Place GFSDP leftmost for highest priority in the method resolution order
    for module in modules:  # noqa: PLR1704
        cls = module.__class__
        new_cls = cls_to_fsdp_cls.get(cls, None)
        if not new_cls:
            # ``_gfsdp_orig_cls`` rather than an index into ``__mro__``: with
            # ``FSDPModule`` in the bases the original class is no longer at a
            # fixed position. See ``GFSDPModule.__new__``.
            dct = {
                "__deepcopy__": _unimplemented_deepcopy,
                "_gfsdp_orig_cls": cls,
            }
            new_cls = type(f"GFSDP{cls.__name__}", (GFSDPModule, cls), dct)
            cls_to_fsdp_cls[cls] = new_cls
        module.__class__ = new_cls
        _publish_state_under_torch_contract(module)
    return cast("GFSDPModule", arg_module)


def _publish_state_under_torch_contract(module: nn.Module) -> None:
    """Make ``torch.distributed.fsdp.fully_shard.state(module)`` find our state.

    The pipeline schedule reaches a stage's sharding state through that call
    (``pipelining/stage.py::perform_reduce_grad``) and then walks
    ``_state_ctx.all_states`` and ``_fsdp_param_groups`` -- names our state
    already uses. What it cannot do is find the state at all: a composable
    contract keys its state by its own function, and ``g_fully_shard`` is a
    different contract than ``fully_shard``. Publishing the same state object
    under torch's key bridges that; the dict is the module's own, one entry
    per contract.
    """
    state = _get_module_fsdp_state(module)
    if state is None:
        raise AssertionError(f"{module} was sharded but carries no GFSDP state")
    all_state = module.__dict__.setdefault(_CONTRACT_STATE_KEY, {})
    all_state[_torch_fully_shard.__wrapped__] = state


def _get_mesh_info(mesh: DeviceMesh) -> FSDPMeshInfo | HSDPMeshInfo:
    if mesh.ndim == 1:
        return FSDPMeshInfo(mesh, shard_mesh_dim=0)
    if mesh.ndim == 2:
        return HSDPMeshInfo(mesh, replicate_mesh_dim=0, shard_mesh_dim=1)
    raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")


def _partition_params(
    params: list[nn.Parameter],
    param_meshes: Sequence[tuple[DeviceMesh, set[nn.Parameter]]] | None,
) -> list[tuple[DeviceMesh | None, list[nn.Parameter]]]:
    """Partition managed parameters into one group per sharding mesh.

    A ``None`` mesh means the unit's own. That group comes first, and order
    within a group follows ``params``, which fixes the sharded layout.

    Args:
        params: The parameters this unit manages, in sharding order.
        param_meshes: Parameters that shard over a mesh other than the unit's.

    Returns:
        One ``(mesh, params)`` entry per group, empty groups dropped.

    Raises:
        ValueError: A listed parameter is not managed by this unit, or is
            listed under more than one mesh.
    """
    if not param_meshes:
        return [(None, params)] if params else []

    managed_params = set(params)
    claimed: set[nn.Parameter] = set()
    extra_partitions: list[tuple[DeviceMesh | None, list[nn.Parameter]]] = []
    for param_mesh, mesh_params in param_meshes:
        if unmanaged := mesh_params - managed_params:
            raise ValueError(
                f"param_meshes lists {len(unmanaged)} parameter(s) this fully_shard "
                "call does not manage: they are outside the module tree, ignored, "
                "or already owned by a nested unit."
            )
        if overlap := mesh_params & claimed:
            raise ValueError(
                f"param_meshes assigns {len(overlap)} parameter(s) to more than one mesh."
            )
        claimed |= mesh_params
        if group_params := [param for param in params if param in mesh_params]:
            extra_partitions.append((param_mesh, group_params))

    partitions: list[tuple[DeviceMesh | None, list[nn.Parameter]]] = []
    if default_params := [param for param in params if param not in claimed]:
        partitions.append((None, default_params))
    partitions.extend(extra_partitions)
    return partitions


def _unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError(
        "GFSDP does not support deepcopy. Please use state dict for serialization."
    )


def _iter_fsdp_param_groups(
    module: nn.Module,
    recurse: bool,
) -> Iterator[GFSDPParamGroup]:
    """Yield unique GFSDP parameter groups in the selected module subtree."""
    modules = list(module.modules()) if recurse else [module]
    seen_param_groups: set[GFSDPParamGroup] = set()
    for child_module in modules:
        if not isinstance(child_module, GFSDPModule):
            continue
        state = child_module._get_fsdp_state()
        for param_group in state._fsdp_param_groups:
            if param_group in seen_param_groups:
                continue
            seen_param_groups.add(param_group)
            yield param_group


class GFSDPModule(FSDPModule):
    """The GFSDP half of a sharded module's dynamic class.

    Inherits ``FSDPModule`` to be recognized by ``isinstance``:
    ``torch.distributed.pipelining`` decides that way whether a stage needs
    data-parallel handling, and a duck-typed class fails those gates silently
    -- no error, just a schedule that reduce-scatters on every microbatch and
    never calls the stage's post-backward. Every method of the protocol is
    overridden below, so the inherited implementations are never reached; the
    base class is there to be recognized.

    ``ChunkedCELoss`` gates its whole chunked protocol on the same
    ``isinstance``, so dropping this base silently un-coalesces the lm_head's
    per-chunk gradient sync as well.

    Recognition alone is not enough under a pipeline: the schedule turns
    gradient sync off for a whole stage at once, and a single shared gradient
    buffer allows that for one parameter group at a time. See
    ``set_allow_no_grad_reduce``.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        # The class this one was swapped in for, recorded when the dynamic
        # class was built. Not an index into ``__mro__``: ``FSDPModule`` sits
        # in there too now, and the original is no longer at a fixed position.
        orig_cls = cls.__dict__.get("_gfsdp_orig_cls")
        if orig_cls is None:
            raise AssertionError(
                f"{cls} has no recorded original class; it was not built by "
                "g_fully_shard"
            )
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)
        return self

    def set_allow_no_grad_reduce(self, enable: bool) -> None:
        """Give every parameter group its own gradient buffer.

        One buffer shared by the whole model is the default, and it is why
        only one group at a time may hold an unreduced gradient
        (``validate_shared_state``). A pipeline schedule breaks that rule by
        construction: it turns gradient sync off for every group of a stage
        for all microbatches but the last. Per-group buffers lift the
        restriction and cost memory -- the sum of the groups instead of the
        largest.

        Must be called before the first forward, which is when the root state
        allocates the buffers.
        """
        state = self._get_fsdp_state()
        if state._is_root is not None:
            raise RuntimeError(
                "set_allow_no_grad_reduce must be called before the first "
                "forward: the buffers it chooses between are allocated in "
                "lazy init"
            )
        state._allow_no_grad_reduce = enable

    def set_own_grad_buffer(self) -> None:
        """Give this unit a gradient buffer no other unit shares.

        The default is one gradient buffer per process group, so exactly one
        unit at a time may hold an unreduced gradient. A unit that suppresses
        its gradient sync across several backward calls holds it for that whole
        window, and the next unit to reach ``post_backward`` raises. A buffer of
        its own lets it hold as long as it likes while everyone else keeps
        cycling through the shared one.

        Costs this unit's unsharded gradient, once -- unlike
        ``set_allow_no_grad_reduce``, which gives EVERY unit its own and so
        keeps the whole model's unsharded gradient resident. Reach for that one
        only when every group suppresses sync, which is what a pipeline
        schedule does.

        Must be called before the first forward, which is when the buffers are
        allocated.
        """
        state = self._get_fsdp_state()
        if state._is_root is not None:
            raise RuntimeError(
                "set_own_grad_buffer must be called before the first forward: "
                "the buffer it asks for is allocated in lazy init"
            )
        state._own_grad_buffer = True

    def set_shard_order(self, modules: "list[GFSDPModule]") -> None:
        """Declare the order forward enters these units, for buffer dealing.

        Data buffers are dealt to parameter groups by alternating along a list
        of groups, and a buffer has one owner at a time: with
        ``reshard_after_forward=True`` the owner only releases in its
        post-forward, while the prefetch of the next unit runs in its
        pre-forward. So two units adjacent in forward must land on different
        buffers, and that holds only if the list is walked in forward order.

        Left undeclared, the list comes from ``root.named_modules()`` -- module
        declaration order, which is not forward order for every model. On a
        ``Decoder`` with MTP it is not: ``lm_head`` is declared before ``mtp``
        and runs after it, which put ``norm`` and ``mtp.layers.0`` on one
        buffer and killed every ``reshard_after_forward="always"`` run on step
        one. Declaration order cannot simply be changed to fix it: weights are
        drawn in that order, so moving a submodule moves the model.

        Units this omits keep their traversal position, after the declared
        ones. Must be called before the first forward, which is when the
        buffers are allocated.
        """
        _assert_all_fsdp_modules(modules)
        state = self._get_fsdp_state()
        if state._is_root is not None:
            raise RuntimeError(
                "set_shard_order must be called before the first forward: the "
                "buffers it orders are allocated in lazy init"
            )
        state._shard_order = [module._get_fsdp_state() for module in modules]

    def reshard(self) -> None:
        state = self._get_fsdp_state()
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.reshard()

    def unshard(self, async_op: bool = False) -> "UnshardHandle | None":
        state = self._get_fsdp_state()
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.unshard(async_op=async_op)
        handle = _UnshardHandleImpl(state._fsdp_param_groups)
        if async_op:
            return handle
        handle.wait()
        return None

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        state = self._get_fsdp_state()
        state._state_ctx.is_last_backward = is_last_backward

    def set_requires_gradient_sync(
        self, requires_gradient_sync: bool, *, recurse: bool = True
    ) -> None:
        self_module = cast("nn.Module", self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, GFSDPModule):
                state = module._get_fsdp_state()
                if not state._fsdp_param_groups:
                    continue
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.reduce_grads = requires_gradient_sync
                state.validate_shared_state()

    def set_reshard_after_forward(
        self, reshard_after_forward: bool, recurse: bool = True
    ) -> None:
        if not isinstance(reshard_after_forward, bool):
            raise ValueError(
                f"reshard_after_forward should be a bool, got {type(reshard_after_forward)}"
            )
        self_module = cast("nn.Module", self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, GFSDPModule):
                state = module._get_fsdp_state()
                state._auto_reshard_after_forward = False
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.post_forward_mesh_info = (
                        _get_post_forward_mesh_info(
                            reshard_after_forward, fsdp_param_group.mesh_info
                        )
                    )

    def set_reshard_after_backward(
        self, reshard_after_backward: bool, *, recurse: bool = True
    ) -> None:
        self_module = cast("nn.Module", self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, GFSDPModule):
                state = module._get_fsdp_state()
                if not state._fsdp_param_groups:
                    continue
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.reshard_after_backward = reshard_after_backward
                state.validate_shared_state()

    def set_modules_to_forward_prefetch(self, modules: "list[GFSDPModule]") -> None:
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_forward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    def set_modules_to_backward_prefetch(self, modules: "list[GFSDPModule]") -> None:
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_backward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    def set_post_optim_event(self, event: torch.Event) -> None:
        self._get_fsdp_state()._state_ctx.post_optim_event = event

    def set_unshard_in_backward(self, unshard_in_backward: bool) -> None:
        state = self._get_fsdp_state()
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.unshard_in_backward = unshard_in_backward

    def set_gradient_divide_factor(self, factor: float) -> None:
        state = self._get_fsdp_state()
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.gradient_divide_factor = factor

    def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
        state = self._get_fsdp_state()
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.force_sum_reduction_for_comms = enable

    def lazy_init(self) -> None:
        self._get_fsdp_state()._lazy_init()

    def setup_ema_buffer(self, recurse: bool = True) -> None:
        """
        Allocate and register whole-group EMA buffers in this subtree.
        Frozen parameter groups are skipped entirely.
        """
        state_ctx = self._get_fsdp_state()._state_ctx
        for param_group in _iter_fsdp_param_groups(cast("nn.Module", self), recurse):
            if not param_group.requires_grad:
                continue
            param_group.setup_ema_buffer()
            state_ctx._ema_tracked_param_groups.add(param_group)

    def _get_ema_tracked_param_groups(
        self,
        recurse: bool = True,
    ) -> tuple[GFSDPParamGroup, ...]:
        """Return registered EMA groups belonging to this module subtree."""
        tracked_param_groups = (
            self._get_fsdp_state()._state_ctx._ema_tracked_param_groups
        )
        return tuple(
            param_group
            for param_group in _iter_fsdp_param_groups(cast("nn.Module", self), recurse)
            if param_group in tracked_param_groups
        )

    def update_ema_buffer_after_load(
        self, force_update: bool = False, recurse: bool = True
    ) -> None:
        """Initialize registered EMA groups after loading main parameters."""
        for param_group in self._get_ema_tracked_param_groups(recurse):
            param_group.update_ema_buffer_after_load(force_update)

    def update_ema_params(self, alpha: float, recurse: bool = True) -> None:
        """Update all registered whole-group EMA buffers in this subtree."""
        for param_group in self._get_ema_tracked_param_groups(recurse):
            param_group.update_ema_params(alpha)

    def swap_ema_and_weights(self, recurse: bool = True) -> None:
        """Swap main and EMA buffers for registered groups in this subtree."""
        for param_group in self._get_ema_tracked_param_groups(recurse):
            param_group.swap_ema_and_weights()

    def _get_fsdp_state(self) -> GFSDPState:
        if (state := _get_module_fsdp_state(cast("nn.Module", self))) is None:
            raise AssertionError(f"No GFSDP state found on {self}")
        return state

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        # Reshard to ensure that sharded parameters are registered
        self.reshard()
        ret = super()._apply(*args, **kwargs)  # type: ignore[misc]
        state = self._get_fsdp_state()
        with torch.no_grad():
            for fsdp_param_group in state._fsdp_param_groups:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    fsdp_param.reset_sharded_param()
        return ret


class UnshardHandle:
    def wait(self) -> None:
        return


class _UnshardHandleImpl(UnshardHandle):
    def __init__(self, fsdp_param_groups: Sequence[GFSDPParamGroup]):
        self._fsdp_param_groups = list(fsdp_param_groups)
        self._all_gather_results: list[AllGatherResult] = []

    def wait(self) -> None:
        if self._fsdp_param_groups:
            self._all_gather_results = [
                all_gather_result
                for fsdp_param_group in self._fsdp_param_groups
                if (all_gather_result := fsdp_param_group.wait_for_unshard())
                is not None
            ]
            # Avoid keeping a reference
            self._fsdp_param_groups = []
        else:
            for all_gather_result in self._all_gather_results:
                all_gather_result.wait()


def _assert_all_fsdp_modules(modules: Iterable[Any]) -> None:
    for module in modules:
        if not isinstance(module, GFSDPModule):
            raise ValueError(f"Expects GFSDPModule but got {type(module)}: {module}")
