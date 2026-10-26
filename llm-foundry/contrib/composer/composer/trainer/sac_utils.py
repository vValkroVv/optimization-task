from collections import defaultdict
from contextlib import (
    AbstractContextManager,
    nullcontext,
    ExitStack,
    ContextDecorator,
)
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import enum
import sys
import warnings

import torch

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils._python_dispatch import TorchDispatchMode

import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_flatten, tree_map

from composer.trainer.cpu_offload import get_cpu_offload_context
from composer.trainer.torch_utils_checkpoint import checkpoint
from composer.utils.model_utils import get_fsdp_lm_backbone

import logging

log = logging.getLogger(__name__)


NO_RECOMPUTE_NAMESPACE = "norecompute"

cpu_offload_context = nullcontext()
cpu_offload_sync_function = lambda x: x
activation_cpu_offload_enabled = False
_should_offload_current_inputs: bool = False

# Role names that are allowed in offload_inputs_modules config.
_OFFLOAD_INPUTS_MODULES_ALLOWLIST = frozenset(
    {
        "self_attn",
        "block_sparse_moe",
        "post_feedforward_layernorm",
        "decoder_layer",  # full-layer offloading used by the Gigar model
    }
)

# Maps each role name to the implementation classes that carry that role.
# Used for documentation and validation hints; the runtime check uses _cpu_offload_role attribute.
_OFFLOAD_ROLE_CLASS_REGISTRY: dict[str, list[str]] = {
    "self_attn": ["AttnGate"],
    "block_sparse_moe": ["DeepseekGMMMoeBlock"],
    "post_feedforward_layernorm": ["ZeroCenteredGatedNorm", "GatedNorm"],
    "decoder_layer": ["LlamaDecoderLayer"],
}


def count_activation_offload_slots(module: torch.nn.Module) -> int:
    """
    Number of CPU offload forward segments for the subtree rooted at `module`.

    Only modules that define `_cpu_offload_role` are considered (presence
    defines a slot; the attribute value is used separately for offload policy).

    To avoid double-counting parent and child, only "marking leaves" count: marked
    modules such that no strict descendant in this subtree is also marked. This covers
    both inner-block tagging (LlamaMixDecoderLayer) and a single tag on the layer root
    (LlamaDecoderLayer in gigar).
    """
    marked = [m for m in module.modules() if hasattr(m, "_cpu_offload_role")]
    if not marked:
        log.warning(
            "%s: no submodule has _cpu_offload_role; "
            "assuming 1 activation forward slot.",
            module.__class__.__name__,
        )
        return 1

    marked_set = frozenset(marked)
    leaf_slots = 0
    for m in marked:
        if any(d in marked_set for d in m.modules() if d is not m):
            continue
        leaf_slots += 1
    return leaf_slots


def validate_cpu_offload_roles(
    layers_list: list,
    offload_inputs_modules: List[str],
) -> None:
    """Check that each tagged module's class is an approved implementation for its role.

    For every submodule that has ``_cpu_offload_role`` set to a role listed in
    ``offload_inputs_modules``, the module's class name must appear in
    ``_OFFLOAD_ROLE_CLASS_REGISTRY[role]``.  Raises ``ValueError`` on the first
    mismatch so the user gets a clear message before training begins.
    """
    for layer in layers_list:
        for m in layer.modules():
            role = getattr(m, "_cpu_offload_role", None)
            if role is None or role not in offload_inputs_modules:
                continue
            allowed = _OFFLOAD_ROLE_CLASS_REGISTRY.get(role)
            root_module = getattr(m, "root_module", None) or m
            if allowed is not None and root_module.__class__.__name__ not in allowed:
                raise ValueError(
                    f"Module assigned cpu offload role '{role}' is implemented by "
                    f"'{root_module.__class__.__name__}', but only the following classes are "
                    f"allowed for this role: {allowed}. "
                    f"Update _OFFLOAD_ROLE_CLASS_REGISTRY if this is a valid implementation."
                )


def setup_cpu_offload(fsdp_obj, fsdp_config: Dict[str, Any]) -> None:
    """Initialize CPU offload for GigaFSDP."""
    global cpu_offload_context, cpu_offload_sync_function

    layers_list = list(get_fsdp_lm_backbone(fsdp_obj).layers)
    offload_inputs_modules = fsdp_config.get("offload_inputs_modules") or []
    validate_cpu_offload_roles(layers_list, offload_inputs_modules)

    num_layers_to_checkpoint = fsdp_config.get("num_layers_to_checkpoint")
    num_checkpointed = (
        num_layers_to_checkpoint
        if num_layers_to_checkpoint is not None
        else len(layers_list)
    )
    checkpointed_layers = layers_list[:num_checkpointed]

    activation_fwd_slots = sum(
        count_activation_offload_slots(layer) for layer in layers_list
    )

    mtp_block = getattr(get_fsdp_lm_backbone(fsdp_obj), "mtp_block", None)
    if mtp_block is not None:
        activation_fwd_slots += sum(
            count_activation_offload_slots(mtp_layer.decoder_layer)
            if getattr(mtp_layer, "mtp_block_type", None) == "moe"
            else 0
            for mtp_layer in mtp_block.mtp_layers
        )

    checkpointed_fwd_slots = sum(
        count_activation_offload_slots(layer) for layer in checkpointed_layers
    )

    num_non_offload_slots = fsdp_config.get("num_non_offload_slots", 0)
    assert num_non_offload_slots >= 0, (
        f"The value of num_non_offload_slots should not be negative. But it is set {num_non_offload_slots=}"
    )

    num_offload_slots = min(
        checkpointed_fwd_slots, max(0, activation_fwd_slots - 1 - num_non_offload_slots)
    )

    cpu_offload_context, cpu_offload_sync_function = get_cpu_offload_context(
        enabled=True,
        num_layers=num_offload_slots,
        model_layers=activation_fwd_slots,
    )


@torch.library.custom_op(f"{NO_RECOMPUTE_NAMESPACE}::mm", mutates_args=())
def _(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.mm(lhs, rhs)


@torch.library.custom_op(
    f"{NO_RECOMPUTE_NAMESPACE}::fused_all_gather_matmul", mutates_args=()
)
def _(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    gather_dim: int,
    group_name: str,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    return torch.ops.symm_mem.fused_all_gather_matmul(
        A_shard, Bs, gather_dim, group_name
    )


def _get_mm_list():
    """Build the list of matmul ops lazily so that ops registered after import are picked up."""
    ops = [
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
    ]
    for attr_path in (
        "symm_mem.fused_all_gather_matmul.default",
        "symm_mem.fused_matmul_reduce_scatter.default",
    ):
        try:
            namespace, name, overload = attr_path.split(".")
            op = getattr(getattr(getattr(torch.ops, namespace), name), overload)
            ops.append(op)
        except AttributeError:
            pass
    return ops


def _get_always_saved_list():
    """Build the list of always-saved ops lazily so that late-registered ops are included."""
    ops = [
        torch.ops.aten.bmm.default,
    ]
    for attr_path in (
        "flash_attn._flash_attn_forward.default",
        "flash_attn_3.fwd.default",
    ):
        try:
            namespace, name, overload = attr_path.split(".")
            op = getattr(getattr(getattr(torch.ops, namespace), name), overload)
            ops.append(op)
        except AttributeError:
            pass
    return ops


def _get_cpu_offload_list():
    """Build the list of ops that should use CPU offload hooks."""
    return []


class SacStrategy(enum.IntEnum):
    DISABLE = 0  # full activation checkpointing
    LVL1 = 1  # saved matmuls, attn
    LVL2 = 2  # saved matmuls, attn wo gate
    LVL3 = 3  # saved matmuls, attn wo up, gate


BLACK_LIST_REGISTRY = defaultdict(dict)
BLACK_LIST_REGISTRY["LlamaMixDecoderLayer"] = {
    SacStrategy.LVL1: [],
    SacStrategy.LVL2: [3],
    SacStrategy.LVL3: [3, 4],
}
BLACK_LIST_REGISTRY["DeepseekGMMMoeBlock"] = {
    SacStrategy.LVL1: [],
    SacStrategy.LVL2: [3],
    SacStrategy.LVL3: [3, 4],
}
BLACK_LIST_REGISTRY["AttnGate"] = {
    SacStrategy.LVL1: [],
    SacStrategy.LVL2: [3],
    SacStrategy.LVL3: [3, 4],
}
BLACK_LIST_REGISTRY["LlamaDecoderLayer"] = {
    SacStrategy.LVL1: [],
    SacStrategy.LVL2: [3],
    SacStrategy.LVL3: [3, 4],
}


@dataclass
class SacConfig:
    sac_strategy: int = 0
    activation_checkpointing: bool = False
    activation_cpu_offload: bool = False
    layer_name: str = "LlamaMixDecoderLayer"
    success: bool = True
    offload_inputs_modules: Optional[List[str]] = None

    def __post_init__(self):
        if self.sac_strategy not in [0, 1, 2, 3]:
            warnings.warn(
                f"{self.sac_strategy=} this is an incorrect sac_strategy value. The sac_strategy value was set to 0."
            )
            self.sac_strategy = 0

        if self.activation_cpu_offload:
            if not self.offload_inputs_modules:
                raise ValueError(
                    "activation_cpu_offload=True requires offload_inputs_modules to be a non-empty list "
                    f"of role names. Supported roles: {sorted(_OFFLOAD_INPUTS_MODULES_ALLOWLIST)}. "
                    f"Each role maps to implementation classes: {_OFFLOAD_ROLE_CLASS_REGISTRY}."
                )

        if self.offload_inputs_modules is not None:
            unknown = (
                set(self.offload_inputs_modules) - _OFFLOAD_INPUTS_MODULES_ALLOWLIST
            )
            if unknown:
                raise ValueError(
                    "offload_inputs_modules may only contain role names that are "
                    f"supported for activation CPU offload inputs: {sorted(_OFFLOAD_INPUTS_MODULES_ALLOWLIST)}. "
                    f"Unsupported: {sorted(unknown)}"
                )


def create_recompute_context_policy(module):
    class RecomputeStateManager(ContextDecorator):
        def __init__(self, module):
            self.module = module

        def __enter__(self):
            self.module.set_is_recompute_state(True)
            return self

        def __exit__(self, *exc):
            self.module.set_is_recompute_state(False)
            return False

    recompute_context_managers = []
    for submodule in module.modules():
        if hasattr(submodule, "set_is_recompute_state"):
            recompute_context_managers.append(RecomputeStateManager(submodule))

    return (nullcontext(), joined_context_managers(*recompute_context_managers))


def _get_custom_policy(module, sac_config, meta):
    if sac_config.sac_strategy == SacStrategy.DISABLE:
        return create_recompute_context_policy(module)
    layer_name = module.__class__.__name__
    try:
        black_list = BLACK_LIST_REGISTRY[layer_name][sac_config.sac_strategy]
    except KeyError:
        warnings.warn(
            f"Layer {layer_name} has no sac_strategy {sac_config.sac_strategy}. The sac_strategy value was set to 0."
        )
        sac_config.success = False
        return create_recompute_context_policy(module)
    sac_config.success = True

    mm_list = _get_mm_list()
    always_saved_list = _get_always_saved_list()
    cpu_offload_list = _get_cpu_offload_list()

    def _custom_policy(ctx, func, *args, **kwargs):
        if func in mm_list:
            mode = "forward" if not ctx.is_recompute else "recompute"
            mm_count_key = f"{mode}_mm_count"
            meta[mm_count_key] += 1
            to_save = meta[mm_count_key] not in black_list
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )
        elif func in always_saved_list:
            return CheckpointPolicy.MUST_SAVE
        elif sac_config.activation_cpu_offload and func in cpu_offload_list:
            return CheckpointPolicy.CPU_OFFLOAD
        return CheckpointPolicy.PREFER_RECOMPUTE

    return _custom_policy


def get_wrapper_fn(sac_config):
    def custom_checkpoint_wrapper(module: torch.nn.Module) -> torch.nn.Module:
        debug_state = _DebugState(layer_name=module.__class__.__name__)

        def selective_checkpointing_context_fn():
            global _should_offload_current_inputs
            module_offload_role = getattr(module, "_cpu_offload_role", None)
            if sac_config.offload_inputs_modules is None:
                _should_offload_current_inputs = False
            else:
                _should_offload_current_inputs = (
                    module_offload_role is not None
                    and module_offload_role in sac_config.offload_inputs_modules
                )

            meta = defaultdict(int)
            policy_fn = _get_custom_policy(module, sac_config, meta)
            if sac_config.sac_strategy != SacStrategy.DISABLE and sac_config.success:
                policy_fn = create_selective_checkpoint_contexts(
                    policy_fn,
                    debug_state=debug_state,
                    cpu_offload_context=cpu_offload_context,
                )
            return policy_fn

        return checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            checkpoint_fn=checkpoint,
            preserve_rng_state=False,
            use_reentrant=False,
        )

    return custom_checkpoint_wrapper


def apply_ac(obj, fsdp_obj, obj_name, fsdp_config):
    global activation_cpu_offload_enabled
    activation_cpu_offload_enabled = fsdp_config["activation_cpu_offload"]

    if (
        not fsdp_config["activation_checkpointing"]
        and not fsdp_config["activation_cpu_offload"]
    ):
        return

    if not fsdp_config["gigafsdp_enabled"]:
        wrapper_fn, _check_fn = get_fsdp_sac(obj, fsdp_config)
    else:
        sac_config = SacConfig(
            activation_checkpointing=fsdp_config["activation_checkpointing"],
            sac_strategy=fsdp_config["sac_strategy"],
            activation_cpu_offload=fsdp_config["activation_cpu_offload"],
            offload_inputs_modules=fsdp_config.get("offload_inputs_modules", None),
        )

        if not sac_config.activation_checkpointing:
            return

        auto_wrap_policy = fsdp_config["gigafsdp_wrappers"][obj_name][
            "activation_checkpointing_auto_wrap_policy"
        ]

        if fsdp_config["activation_cpu_offload"]:
            setup_cpu_offload(fsdp_obj, fsdp_config)

        if auto_wrap_policy is None:
            raise ValueError(
                f"activation_checkpointing_auto_wrap_policy is None for {obj_name},"
                + " but activation_checkpointing is enabled"
            )

        _register_sac_modes_with_triton_hop()
        wrapper_fn = get_wrapper_fn(sac_config)

    apply_activation_checkpointing(
        fsdp_obj,
        checkpoint_wrapper_fn=wrapper_fn,
        auto_wrap_policy=None
        if not fsdp_config["gigafsdp_enabled"]
        else auto_wrap_policy,
        check_fn=lambda _: True if fsdp_config["gigafsdp_enabled"] else _check_fn,
    )


def get_fsdp_sac(obj, fsdp_config):
    activation_checkpointing = fsdp_config["activation_checkpointing"]

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        offload_wrapper,
    )

    if not fsdp_config["activation_checkpointing_reentrant"]:
        first_wrap_fn = lambda m: (
            checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            if fsdp_config["activation_checkpointing"]
            else (lambda module: module)
        )
        second_wrap_fn = (
            (
                lambda module: offload_wrapper(
                    first_wrap_fn(module) if activation_checkpointing else module,  # type: ignore reportGeneralTypeIssues
                )
            )
            if fsdp_config["activation_cpu_offload"]
            else first_wrap_fn
        )
    else:
        first_wrap_fn = (
            checkpoint_wrapper
            if fsdp_config["activation_checkpointing"]
            else (lambda module: module)
        )
        second_wrap_fn = (
            (
                lambda module: offload_wrapper(
                    first_wrap_fn(module) if activation_checkpointing else module
                )  # type: ignore reportGeneralTypeIssues
            )
            if fsdp_config["activation_cpu_offload"]
            else first_wrap_fn
        )

    # Choose which modules to activation checkpoint according to the following priority:
    # If module has attribute `module._activation_checkpointing = ...`, always respect it
    # Otherwise checkpoint if root object `obj.activation_checkpointing_fn(module)` is true
    def _check_fn(module: torch.nn.Module) -> bool:
        if isinstance(module, FullyShardedDataParallel):
            return False
        if hasattr(module, "_activation_checkpointing"):
            return bool(module._activation_checkpointing)
        if hasattr(obj, "activation_checkpointing_fn") and isinstance(
            obj.activation_checkpointing_fn, Callable
        ):
            return obj.activation_checkpointing_fn(module)
        return False

    return second_wrap_fn, _check_fn


class JoinedContextManagers(AbstractContextManager):
    """Reusable context manager that enters multiple context managers.
    Unlike a @contextmanager generator, this can be entered and exited multiple times.
    """

    def __init__(self, *cms):
        self.cms = cms

    def __enter__(self):
        self._stack = ExitStack()
        self._stack.__enter__()
        try:
            self._entered = [self._stack.enter_context(cm) for cm in self.cms]
            return self._entered
        except Exception:
            self._stack.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


def joined_context_managers(*cms):
    """Return a reusable context manager that enters all *cms. Safe to use multiple times."""
    return JoinedContextManagers(*cms)


# ---------------------------------------------------------------------------
# Selective Activation Checkpointing (SAC) — local implementation.
# Mirrors torch.utils.checkpoint (SelectiveCheckpointContext, CheckpointPolicy,
# create_selective_checkpoint_contexts, etc.) for PyTorch version compatibility
# and to avoid relying on non-stable torch APIs. Triton HOP registration below
# is required so SAC modes work under torch.compile with Triton kernels.
# ---------------------------------------------------------------------------


class SelectiveCheckpointContext:
    """
    Context passed to policy function during selective checkpointing.

    This class is used to pass relevant metadata to the policy function during
    selective checkpointing. The metadata includes whether the current invocation
    of the policy function is during recomputation or not.

    Example:
        >>> # xdoctest: +SKIP(stub)
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    print(ctx.is_recompute)
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """

    def __init__(self, *, is_recompute) -> None:
        self.is_recompute = is_recompute


def _policy_from_bool(b):
    # For backward compatibility
    return CheckpointPolicy.MUST_SAVE if b else CheckpointPolicy.PREFER_RECOMPUTE


SAC_IGNORED_OPS = {
    # AC inserts different number of detach during forward and recompute.
    torch.ops.aten.detach.default,
    # AC's determinism check invokes additional metadata ops during forward.
    # With subclasses involved, these metadata ops become dispatchable, this
    # can result in incorrectness if these ops are selected cached.
    torch.ops.prim.device.default,
    # View / shape ops: never mutate data, O(1), always cheap to recompute.
    torch.ops.aten.t.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.select.int,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.alias.default,
} | set(torch._subclasses.functional_tensor.FunctionalTensor.metadata_fns)  # type: ignore[has-type]


class CheckpointPolicy(enum.Enum):
    """
    Enum for specifying the policy for checkpointing during backpropagation.

    The following policies are supported:

    - ``{MUST,PREFER}_SAVE``: The operation's output will be saved during the forward
      pass and will not be recomputed during the backward pass
    - ``{MUST,PREFER}_RECOMPUTE``: The operation's output will not be saved during the
      forward pass and will be recomputed during the backward pass

    Use ``MUST_*`` over ``PREFER_*`` to indicate that the policy should not be overridden
    by other subsystems like `torch.compile`.

    .. note::
        A policy function that always returns ``PREFER_RECOMPUTE`` is
        equivalent to vanilla checkpointing.

        A policy function that returns ``PREFER_SAVE`` every op is
        NOT equivalent to not using checkpointing. Using such a policy would
        save additional tensors not limited to ones that are actually needed for
        gradient computation.
    """

    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3
    CPU_OFFLOAD = 4


def _is_compiling(func, args, kwargs):
    # Check if we are under AOTAutograd tracing
    # Checking that a functional mode is active should always do what we want
    return torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is not None


class _VersionWrapper:
    # Check that cached tensors are not mutated.
    def __init__(self, val) -> None:
        self.val: Union[torch.Tensor, Any] = val
        self.version: Optional[int] = (
            val._version if isinstance(val, torch.Tensor) else None
        )

    def get_val(self, allow_cache_entry_mutation):
        if self.version is not None and not allow_cache_entry_mutation:
            if self.val._version != self.version:
                # Can we give user a stack trace of where the mutation happened?
                raise RuntimeError(
                    "Tensor cached during selective activation checkpoint has been mutated"
                )
        return self.val


class _DebugState:
    """Per-layer state for first-iteration debug checks.

    Two independent flags control the two dispatch modes:
      - ``caching_enabled``  — mutation checks inside _CachingTorchDispatchMode
      - ``cached_enabled``   — leftover-storage checks inside _CachedTorchDispatchMode
    """

    def __init__(self, layer_name: str = "") -> None:
        self.caching_enabled = True
        self.cached_enabled = True
        self.layer_name = layer_name

    @property
    def enabled(self):
        """Backward-compatible alias used by _CachingTorchDispatchMode."""
        return self.caching_enabled

    def disable(self) -> None:
        """Disable _CachingTorchDispatchMode checks (called from its __exit__)."""
        self.caching_enabled = False

    def disable_cached(self) -> None:
        """Disable _CachedTorchDispatchMode checks (called from its __exit__)."""
        self.cached_enabled = False


def _check_last_saved(
    prev_saved: Tuple[Any, Any, Any],
    current_func: Any,
    layer_name: str = "",
) -> None:
    """Verify that the last saved activation was not mutated in-place."""
    saved_func, saved_entry, clone_entry = prev_saved
    flat_entries, _ = tree_flatten(saved_entry)
    flat_clones, _ = tree_flatten(clone_entry)
    for j, (wrapper, cloned) in enumerate(zip(flat_entries, flat_clones)):
        if isinstance(wrapper, _VersionWrapper) and isinstance(
            wrapper.val, torch.Tensor
        ):
            if not torch.equal(wrapper.val, cloned):
                raise RuntimeError(
                    f"[SAC DEBUG] Storage mutation detected in layer '{layer_name}'! "
                    f"Tensor saved by {saved_func} (element {j}) was modified in-place. "
                    f"Detected at {current_func}."
                )


def _maybe_detach(x, any_ret_has_alias_info):
    # We detach for two separate reasons:
    # - For view ops, we need to ensure that when the tensor is returned from
    #   CachedDispatchMode, as_view sees that the AutogradMeta is nullptr
    # - Avoid reference cycles
    # For case 1, it is not enough to check whether x has differentiable dtype
    # because non-differentiable dtype can have non-nullptr AutogradMeta, e.g.
    # when the tensor is a view.
    if isinstance(x, torch.Tensor) and (
        x.is_floating_point() or x.is_complex() or any_ret_has_alias_info
    ):
        with torch._C._SetExcludeDispatchKeyGuard(
            torch._C.DispatchKey.ADInplaceOrView, False
        ):
            # Ensure that view performed beneath autograd properly propagates
            # version counter. TODO: Use reentrant_dispatch instead of
            # manually manipulating dispatch keys. Using reentrant_dispatch
            # would respect inference_mode, though that is not relevant for
            # this case.
            x = x.detach()
    return x


class _CachingTorchDispatchMode(TorchDispatchMode):
    @classmethod
    def ignore_compile_internals(cls):
        return True

    # Used together with _CachedTorchDispatchMode to implement SAC.
    def __init__(
        self,
        policy_fn: Callable,
        storage: Dict[Any, List[Any]],
        cpu_storage=None,
        pack_hook=lambda x: x,
        debug_state: Optional["_DebugState"] = None,
    ) -> None:
        self.policy_fn = policy_fn
        self.storage = storage
        self.debug_state = debug_state
        self._prev_saved: Optional[Tuple] = None
        self.cpu_storage = cpu_storage
        self.pack_hook = pack_hook
        self.use_offload = cpu_storage is not None

    def __exit__(self, *args):
        if self.debug_state is not None:
            self.debug_state.disable()
        self._prev_saved = None
        return super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs

        policy = self.policy_fn(
            SelectiveCheckpointContext(is_recompute=False), func, *args, **kwargs
        )
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        is_compiling = _is_compiling(func, args, kwargs)

        if is_compiling:
            # Overwrite each node's "recompute" tag to add in the user annotation.
            fx_traceback.current_meta["recompute"] = policy

        out = func(*args, **kwargs)

        if (
            self._prev_saved is not None
            and self.debug_state is not None
            and self.debug_state.enabled
        ):
            _check_last_saved(
                self._prev_saved,
                current_func=func,
                layer_name=self.debug_state.layer_name,
            )

        # HOPs don't support func._schema
        # HOPs don't alias -> this is always true today and will be always true for a long time
        # TODO HOPs don't mutate -> this is always true today but will not be true forever
        if isinstance(func, torch._ops.HigherOrderOperator):
            any_ret_has_alias_info = False
        else:
            any_ret_has_alias_info = any(
                ret.alias_info is not None for ret in func._schema.returns
            )

        if (
            policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE)
            or is_compiling
        ):
            saved_entry = tree_map(
                lambda x: _VersionWrapper(_maybe_detach(x, any_ret_has_alias_info)), out
            )
            self.storage[func].append(saved_entry)
            if self.debug_state is not None and self.debug_state.enabled:
                clone_entry = tree_map(
                    lambda x: (
                        x.val.clone()
                        if isinstance(x, _VersionWrapper)
                        and isinstance(x.val, torch.Tensor)
                        else x
                    ),
                    saved_entry,
                )
                self._prev_saved = (func, saved_entry, clone_entry)
        elif policy == CheckpointPolicy.CPU_OFFLOAD and self.use_offload:
            storage = []
            if isinstance(out, (list, tuple)):
                for t in out:
                    if t is not None:
                        t.activation_offloading = True
                    storage.append(self.pack_hook(t))
            else:
                if out is not None:
                    out.activation_offloading = True
                storage.append(self.pack_hook(out))
            self.cpu_storage[func].append(storage)
        return out


class _CachedTorchDispatchMode(TorchDispatchMode):
    @classmethod
    def ignore_compile_internals(cls):
        return True

    # Used together with _CachingTorchDispatchMode to implement SAC.
    def __init__(
        self,
        policy_fn: Callable,
        storage: Dict[Any, List[Any]],
        allow_cache_entry_mutation: bool,
        verification_storage: Optional[Dict[Any, List[Any]]] = None,
        debug_state: Optional["_DebugState"] = None,
        cpu_storage=None,
        unpack_hook=lambda x: x,
    ) -> None:
        self.policy_fn = policy_fn
        self.storage = storage
        self.allow_cache_entry_mutation = allow_cache_entry_mutation
        self.verification_storage = verification_storage
        self.debug_state = debug_state
        self.last_func: Optional[Any] = None
        self.cpu_storage = cpu_storage
        self.unpack_hook = unpack_hook
        self.use_offload = cpu_storage is not None

    def __exit__(self, *args):
        if self.debug_state is not None and self.debug_state.cached_enabled:
            self.debug_state.disable_cached()
            leftover = {
                func: entries for func, entries in self.storage.items() if entries
            }
            if leftover:
                details = ", ".join(
                    f"{func} ({len(entries)} remaining)"
                    for func, entries in leftover.items()
                )
                log.warning(
                    "Activations saved during forward were not consumed during "
                    "recompute — these ops are being saved but never recomputed: %s",
                    details,
                )
        return super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs

        policy = self.policy_fn(
            SelectiveCheckpointContext(is_recompute=True), func, *args, **kwargs
        )
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        is_compiling = _is_compiling(func, args, kwargs)

        if (
            policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE)
            or is_compiling
        ):
            storage_list = self.storage.get(func)
            if storage_list is None:
                raise RuntimeError(
                    f"{func} encountered during backward, but not found in storage"
                )
            if len(storage_list) == 0:
                raise RuntimeError(
                    "Trying to backward an extra time. You are only allowed to backward once "
                    "on any region computed under selective activation checkpoint."
                )
            out = tree_map(
                lambda x: x.get_val(self.allow_cache_entry_mutation),
                storage_list.pop(0),
            )
        elif policy == CheckpointPolicy.CPU_OFFLOAD and self.use_offload:
            cpu_storage = self.cpu_storage.get(func).pop(0)
            if len(cpu_storage) == 1:
                out = self.unpack_hook(cpu_storage.pop(0))
            elif cpu_storage[0] is None:
                # If recv_x is None, then you need to recalculate the activation
                out = func(*args, **kwargs)
            else:
                out = list()
                for t in cpu_storage:
                    out.append(self.unpack_hook(t))
                out = tuple(out)
        else:
            out = func(*args, **kwargs)

        self.last_func = func
        return out


def create_selective_checkpoint_contexts(
    policy_fn_or_list: Union[Callable, List[Any]],
    allow_cache_entry_mutation: bool = False,
    debug_state: Optional["_DebugState"] = None,
    cpu_offload_context=None,
):
    """
    Helper to avoid recomputing certain ops during activation checkpointing.

    Use this with `torch.utils.checkpoint.checkpoint` to control which
    operations are recomputed during the backward pass.

    Args:
        policy_fn_or_list (Callable or List):
          - If a policy function is provided, it should accept a
            :class:`SelectiveCheckpointContext`, the :class:`OpOverload`, args and
            kwargs to the op, and return a :class:`CheckpointPolicy` enum value
            indicating whether the execution of the op should be recomputed or not.
          - If a list of operations is provided, it is equivalent to a policy
            returning `CheckpointPolicy.MUST_SAVE` for the specified
            operations and `CheckpointPolicy.PREFER_RECOMPUTE` for all other
            operations.
        allow_cache_entry_mutation (bool, optional): By default, an error is
            raised if any tensors cached by selective activation checkpoint are
            mutated in order to ensure correctness. If set to `True`, this check
            is disabled.
        debug_state (_DebugState, optional): If provided and enabled, clones of
            saved activations are stored and verified for in-place mutations
            on the first iteration only. Used internally by get_wrapper_fn.
    Returns:
        A tuple of two context managers.

    Example:
        >>> # xdoctest: +REQUIRES(LINUX)
        >>> import functools
        >>>
        >>> x = torch.rand(10, 10, requires_grad=True)
        >>> y = torch.rand(10, 10, requires_grad=True)
        >>>
        >>> ops_to_save = [
        >>>    torch.ops.aten.mm.default,
        >>> ]
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    if op in ops_to_save:
        >>>        return CheckpointPolicy.MUST_SAVE
        >>>    else:
        >>>        return CheckpointPolicy.PREFER_RECOMPUTE
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> # or equivalently
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, ops_to_save)
        >>>
        >>> def fn(x, y):
        >>>     return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
    # NB: If grad_mode is disabled, checkpoint would not run forward under
    #     context_fn anyway, so proceed as usual.
    if isinstance(policy_fn_or_list, list):
        for op in policy_fn_or_list:
            if not isinstance(
                op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
            ):
                _extra_msg = (
                    (
                        "Please update the OpOverloadPacket to a specific OpOverload."
                        "For example, if you have `torch.ops.aten.mm`, change it to `torch.ops.aten.mm.default`."
                    )
                    if isinstance(op, torch._ops.OpOverloadPacket)
                    else ""
                )
                raise ValueError(
                    f"Expected op in `op_list` to be an OpOverload but got: {op} "
                    f"of type {type(op)}. {_extra_msg}"
                )

        def policy_fn(ctx, op, *args, **kwargs):
            if op in policy_fn_or_list:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE
    elif callable(policy_fn_or_list):
        policy_fn = policy_fn_or_list
    else:
        raise TypeError("policy_fn_or_list must be either a function or a list of ops.")

    storage: Dict[Any, List[Any]] = defaultdict(list)

    if cpu_offload_context is not None and hasattr(cpu_offload_context, "get_hooks"):
        cpu_storage = defaultdict(list)
        pack_hook, unpack_hook = cpu_offload_context.get_hooks()
        return (
            _CachingTorchDispatchMode(
                policy_fn,
                storage,
                debug_state=debug_state,
                cpu_storage=cpu_storage,
                pack_hook=pack_hook,
            ),
            _CachedTorchDispatchMode(
                policy_fn,
                storage,
                allow_cache_entry_mutation,
                cpu_storage=cpu_storage,
                unpack_hook=unpack_hook,
                debug_state=debug_state,
            ),
        )
    return (
        _CachingTorchDispatchMode(
            policy_fn,
            storage,
            debug_state=debug_state,
        ),
        _CachedTorchDispatchMode(
            policy_fn,
            storage,
            allow_cache_entry_mutation,
            debug_state=debug_state,
        ),
    )


_sac_triton_hop_registered = False


def _register_sac_modes_with_triton_hop():
    # Register our SAC modes with Triton HOPs so they don't raise "no rule registered".
    # Uses HOP.py_impl(mode) directly so we don't depend on redirect_to_mode (not in older PyTorch).
    global _sac_triton_hop_registered
    if _sac_triton_hop_registered:
        return
    try:
        import torch._higher_order_ops.triton_kernel_wrap as _tw

        _registered = []
        for _name in (
            "triton_kernel_wrapper_mutation",
            "triton_kernel_wrapper_functional",
        ):
            _hop = getattr(_tw, _name, None)
            if _hop is None or not hasattr(_hop, "py_impl"):
                continue

            # Same as redirect_to_mode(hop, Mode): register handler that calls our __torch_dispatch__
            def _impl(mode, *args, _hop_obj=_hop, **kwargs):
                return mode.__torch_dispatch__(_hop_obj, [], args, kwargs)

            _hop.py_impl(_CachingTorchDispatchMode)(_impl)
            _hop.py_impl(_CachedTorchDispatchMode)(_impl)
            _registered.append(_name)

        if _registered:
            log.info("SAC registered with Triton HOPs: %s", _registered)
        _sac_triton_hop_registered = True
    except Exception as e:  # noqa: BLE001
        log.warning(
            "Could not register SAC with Triton HOPs (torch.compile with Triton may not work): %s",
            e,
        )
