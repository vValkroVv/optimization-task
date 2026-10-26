# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import threading
import uuid
import warnings
import weakref
from collections import defaultdict
from typing import *  # noqa: F403
from typing_extensions import Self
from weakref import ReferenceType

import torch
from torch.utils._pytree import tree_map
from torch.utils._python_dispatch import TorchDispatchMode

from typing import NoReturn

# Lazy import: _get_current_function_mode_stack / DeviceContext appeared in
# newer torch versions. Guard so that the module still imports on older torch.
try:
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._device import DeviceContext
    _HAS_DEVICE_CTX = True
except ImportError:
    _HAS_DEVICE_CTX = False
    DeviceContext = None  # type: ignore[assignment,misc]

# Detect whether torch.random.fork_rng accepts a device_type keyword.
import inspect as _inspect
_fork_rng_supports_device_type = "device_type" in _inspect.signature(torch.random.fork_rng).parameters
del _inspect


__all__ = [
    "checkpoint",
    "CheckpointError",
    "get_device_states",
    "set_device_states",
    "noop_context_fn",
    "set_checkpoint_early_stop",
    "DefaultDeviceType",
    "set_checkpoint_debug_enabled",
    "GraphExecGroup",
    "is_recomputing",
]

_DEFAULT_DETERMINISM_MODE = "default"

# Thread-local: the recompute runs on the autograd worker thread that executes this layer's
# backward, so a concurrent forward on another thread must NOT observe this flag as True.
_recompute_state = threading.local()


def is_recomputing() -> bool:
    """True while a checkpointed forward is re-executing for backward, on THIS thread.

    Lets deterministic, expensive, gradient-free work (e.g. DSA top-k selection) cache its
    result on the real forward and replay it during recompute instead of recomputing it.
    """
    return getattr(_recompute_state, "active", False)

_checkpoint_debug_enabled: bool | None = None
_fallback_graph_exec_group = None


def _get_graph_exec_group():
    if hasattr(torch._C, "_get_graph_exec_group"):
        return torch._C._get_graph_exec_group()
    return _fallback_graph_exec_group


def _set_graph_exec_group(group) -> None:
    global _fallback_graph_exec_group
    if hasattr(torch._C, "_set_graph_exec_group"):
        torch._C._set_graph_exec_group(group)
        return
    _fallback_graph_exec_group = group


@contextlib.contextmanager
def set_checkpoint_debug_enabled(enabled: bool | None):
    """
    Context manager that sets whether checkpoint should print additional debug
    information when running. See the ``debug`` flag for
    :func:`~torch.utils.checkpoint.checkpoint` for more information. Note that
    when set, this context manager overrides the value of ``debug`` passed to
    checkpoint. To defer to the local setting, pass ``None`` to this context.

    Args:
        enabled (bool): Whether checkpoint should print debug information.
            Default is 'None'.
    """
    global _checkpoint_debug_enabled
    try:
        prev = _checkpoint_debug_enabled
        _checkpoint_debug_enabled = enabled
        yield
    finally:
        _checkpoint_debug_enabled = prev


def _get_device_module(device="cuda"):
    if device == "meta":
        return torch.device("meta")
    device_module = getattr(torch, device)
    return device_module


class DefaultDeviceType:
    r"""
    A class that manages the default device type for checkpointing.

    If no non-CPU tensors are present, the default device type will
    be used. The default value is 'cuda'. The device type is used in
    the checkpointing process when determining which device states
    to save and restore for recomputation.
    """

    _default_device_type: str | None = None

    @staticmethod
    def set_device_type(device: str = "cuda") -> None:
        """
        Set the default device type for checkpointing.

        Args:
            device (str): The device type to be set as default. Default is 'cuda'.
        """
        DefaultDeviceType._default_device_type = device

    @staticmethod
    def get_device_type() -> str:
        """
        Get the current default device type for checkpointing.

        Returns:
            str: The current default device type.
        """
        if not DefaultDeviceType._default_device_type:
            # Prefer CUDA when available. torch.accelerator is a newer API
            # that may not exist in older torch versions; fall back safely.
            if torch.cuda.is_available():
                DefaultDeviceType._default_device_type = "cuda"
            else:
                try:
                    acc = torch.accelerator.current_accelerator(True)
                    DefaultDeviceType._default_device_type = acc.type if acc else "cpu"
                except AttributeError:
                    DefaultDeviceType._default_device_type = "cpu"
        return DefaultDeviceType._default_device_type


def _infer_device_type(*args):
    device_types = []

    def add_device_types(arg) -> None:
        nonlocal device_types
        if isinstance(arg, torch.Tensor) and arg.device.type != "cpu":
            device_types.append(arg.device.type)
    tree_map(add_device_types, args)

    device_types_set = set(device_types)
    if len(device_types_set) > 1:
        warnings.warn(
            "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices. "
            "Device state will only be saved for devices of a single device type, and the remaining "
            "devices will be ignored. Consequently, if any checkpointed functions involve randomness, "
            "this may result in incorrect gradients. (Note that if CUDA devices are among the devices "
            "detected, it will be prioritized; otherwise, the first device encountered will be selected.)"
            f"\nDevice types: {sorted(device_types_set)} first device type: {device_types[0]}", stacklevel=2
        )
    if len(device_types) == 0:
        return DefaultDeviceType.get_device_type()
    elif "cuda" in device_types_set:
        return "cuda"
    else:
        return device_types[0]


def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]:
    fwd_device_ids = []

    def add_device_ids(arg) -> None:
        nonlocal fwd_device_ids
        if isinstance(arg, torch.Tensor) and arg.device.type not in {"cpu", "meta"}:
            fwd_device_ids.append(arg.get_device())
    tree_map(add_device_ids, args)

    fwd_device_states = []
    device_module = _get_device_module(_infer_device_type(*args))
    for device_id in fwd_device_ids:
        with device_module.device(device_id):
            fwd_device_states.append(device_module.get_rng_state())

    return fwd_device_ids, fwd_device_states


def set_device_states(devices, states, *, device_type=None) -> None:
    """Sets random number generator states for the specified devices.

    Args:
        devices: Device ids to set states for.
        states: States to set.
        device_type: ``device_type`` of the devices to set states for. Default
            is the device returned by a call to ``DefaultDeviceType.get_device_type()``,
            which is ``cuda`` if not changed by calling ``DefaultDeviceType::set_device_type()``.
    """
    if device_type is None:
        device_type = DefaultDeviceType.get_device_type()
    if device_type == "meta":
        return
    device_module = _get_device_module(device_type)
    for device, state in zip(devices, states):
        with device_module.device(device):
            device_module.set_rng_state(state)


def _is_autocast_available(device_type: str) -> bool:
    """Check if autocast is available for a device type, compatible with older torch."""
    try:
        return torch.amp.is_autocast_available(device_type)
    except AttributeError:
        return device_type == "cuda"


def _is_autocast_enabled(device_type: str) -> bool:
    """Check if autocast is enabled, compatible with older torch."""
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:
        return torch.is_autocast_enabled() if device_type == "cuda" else False


def _get_autocast_dtype_safe(device_type: str):
    """Get autocast dtype for a device type, compatible with older torch."""
    try:
        return torch.get_autocast_dtype(device_type)
    except (AttributeError, TypeError):
        if device_type == "cuda":
            try:
                return torch.get_autocast_gpu_dtype()
            except AttributeError:
                return torch.float16
        return torch.float32


def _get_autocast_kwargs(device_type="cuda"):
    if _is_autocast_available(device_type):
        device_autocast_kwargs = {
            "enabled": _is_autocast_enabled(device_type),
            "dtype": _get_autocast_dtype_safe(device_type),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
    else:
        device_autocast_kwargs = None

    cpu_autocast_kwargs = {
        "enabled": _is_autocast_enabled("cpu"),
        "dtype": _get_autocast_dtype_safe("cpu"),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    return device_autocast_kwargs, cpu_autocast_kwargs


def noop_context_fn():
    return contextlib.nullcontext(), contextlib.nullcontext()


# Note: [torch.compile and checkpoint]
# TorchDynamo does not step inside utils.checkpoint function.  The flow
# looks likes this
#  1) TorchDynamo tries to wrap utils.checkpoint in a HigherOrderOp by
#     speculatively checking if the forward function is safe to trace.
#  2) If yes, then Dynamo-generated Fx graph has the wrapped higher
#     order op. As a result, TorchDynamo does not look inside utils.checkpoint.
#  3) If not, then TorchDynamo falls back to eager by performing a graph
#     break. And here, the following disable wrapper ensures that
#     TorchDynamo does not trigger again on the frames created by
#     utils.checkpoint innards.
@torch._disable_dynamo
def checkpoint(
    function,
    *args,
    use_reentrant: bool | None = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    **kwargs
):
    r"""Checkpoint a model or part of the model (non-reentrant implementation only).

    Activation checkpointing is a technique that trades compute for memory.
    By default, tensors computed during the forward pass are kept alive until
    they are used in gradient computations in the backward pass. To reduce this
    memory usage, tensors produced in the passed :attr:`function` are not kept
    alive until the backward pass. Instead, any passed tensors in :attr:`args`
    are kept alive, and the unsaved tensors are recomputed by re-invoking
    :attr:`function` in the backward pass as needed for gradient computation.

    This implementation supports only ``use_reentrant=False``. Passing
    ``use_reentrant=True`` will emit a warning and fall back to the
    non-reentrant path.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model.
        args: tuple containing inputs to the :attr:`function`

    Keyword args:
        preserve_rng_state(bool, optional): Omit stashing and restoring
            the RNG state during each checkpoint. Default: ``True``
        use_reentrant(bool): Must be ``False`` (or ``None``). Passing ``True``
            will emit a deprecation warning and be treated as ``False``.
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
        determinism_check(str, optional): ``"default"`` or ``"none"``.
            Default: ``"default"``.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation. Default: ``False``.
        early_stop(bool, optional): If ``True``, non-reentrant checkpoint stops
            recomputation as soon as it has computed all needed Tensors.
            Default: ``True``.

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    if use_reentrant is True:
        warnings.warn(
            "composer.trainer.torch_utils_checkpoint: use_reentrant=True is not "
            "supported in this implementation. Falling back to use_reentrant=False.",
            stacklevel=2,
        )

    preserve = kwargs.pop("preserve_rng_state", True)

    gen = _checkpoint_without_reentrant_generator(
        function, preserve, context_fn, determinism_check, debug, early_stop, *args, **kwargs
    )
    # Runs pre-forward logic
    next(gen)
    ret = function(*args, **kwargs)
    # Runs post-forward logic
    try:
        next(gen)
    except StopIteration:
        return ret


def _internal_assert(cond) -> None:
    if not cond:
        raise AssertionError(
            "Something went unexpectedly wrong in activation checkpoint. "
            "Please report this bug by filing an issue to PyTorch."
        )


# NOTE [ Nestable Checkpoint ]
#
# The semantics of nested checkpoint can be defined by two basic rules.
# Following the two rules leads to an important implication that is central
# to motivating the design.
#
# Rule 1. Saved tensors are managed by inner-most checkpoint only and hidden
#         from any outer layers of checkpoint.
#
# Rule 2. The inputs of inner checkpoints are treated as tensors saved to its
#         parent checkpoint.
#
# Implication: To recompute any given saved tensor, we need to recompute all of
#              the checkpoints wrapping it.
#
# Rule 3. We should start recomputation as if there are no checkpoints currently
#         active. Checkpoints encountered during recomputation are still
#         respected.
#
# Rule 4. Lifetime of recomputed tensors
#
#         Recomputed tensors are considered specific to particular invocations
#         of backward and are always cleared immediately as they are unpacked.
#         Particularly, we require this to happen even if retain_graph=True.
#
# Rule 5. Stop recomputation as soon as we've recomputed the saved tensors we
#         know we need.
#
# Rule 6. We support doing backward inside checkpoint context.

_enable_checkpoint_early_stop: bool | None = None


@contextlib.contextmanager
def set_checkpoint_early_stop(enable: bool):
    """Context manager that sets whether checkpoint should stop recomputation early.

    By default, non-reentrant checkpoint stops recomputation as soon as it
    has computed all needed Tensors. This context manager can be used to disable
    that feature if it is problematic for your specific application.

    This context manager only needs to be active when forward is run. It does
    not need to be active during backward.
    """
    global _enable_checkpoint_early_stop
    try:
        prev = _enable_checkpoint_early_stop
        _enable_checkpoint_early_stop = enable
        yield
    finally:
        _enable_checkpoint_early_stop = prev


class _Handle:
    pass


class _Holder:
    def __init__(self) -> None:
        self.handles: dict[int, _Handle | None] = {}


class _NoopSaveInputs(torch.autograd.Function):
    @staticmethod
    def forward(*args):
        return torch.empty((0,))

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        tensor_inputs = [(i, o) for i, o in enumerate(inputs) if isinstance(o, torch.Tensor)]
        tensor_indices, tensors = zip(*tensor_inputs)
        idx2saved_idx = {b: a for a, b in enumerate(tensor_indices)}
        args = [None if isinstance(o, torch.Tensor) else o for o in inputs]

        def get_args(saved_tensors):
            ret = [
                saved_tensors[idx2saved_idx[i]] if i in tensor_indices else o
                for i, o in enumerate(args)
            ]
            return ret[1:]

        ctx.get_args = get_args
        ctx.save_for_backward(*tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise AssertionError("Did not expect to backward on this graph")


def _get_checkpoint_input_offload_hooks():
    try:
        from composer.trainer.sac_utils import (
            cpu_offload_context,
            activation_cpu_offload_enabled,
            _should_offload_current_inputs,
        )
    except ImportError:
        return None

    if not activation_cpu_offload_enabled:
        return None

    if not _should_offload_current_inputs:
        return None

    if not hasattr(cpu_offload_context, "get_hooks"):
        return None
    return cpu_offload_context.get_hooks()


class _CheckpointFrame:
    def __init__(self, recompute_fn, early_stop, unpack_error_cb, metadata_fn) -> None:
        self.recompute_fn = recompute_fn
        self.input_saver = None
        self.weak_holders: List[ReferenceType] = []
        # We store this as a weakkeydictionary so that in the case of a partial
        # backward, the entries in the dict are cleared alongside the Holder
        # which will be removed when the SavedVariable is cleared.
        self.recomputed: DefaultDict[
            int, weakref.WeakKeyDictionary[_Handle, torch.Tensor]
        ] = defaultdict(weakref.WeakKeyDictionary)
        # We need both recomp_counter and recomputed since they can diverge
        # https://github.com/pytorch/pytorch/pull/90105#discussion_r1135889885
        self.recomp_counter: DefaultDict[int, int] = defaultdict(int)
        self.is_recomputed: DefaultDict[int, bool] = defaultdict(bool)

        # See Rule 5
        self.early_stop = early_stop

        # Debugging
        self.metadata_fn = metadata_fn
        self.unpack_error_cb = unpack_error_cb
        self.x_metadatas = []
        self.forward_completed = False
        self.ignore_saved_mismatch = False

        # Metadata for kwargs support: positional args count and the ordered keys
        # of tensor-valued kwargs that were saved alongside positional args.
        # Non-tensor kwargs are NOT stored here – they live in the recompute_fn
        # closure so that GPU memory is not held here.
        self._num_positional: int = 0
        self._tensor_kwargs_keys: tuple = ()

    def save_inputs(self, *args, **tensor_kwargs):
        # tensor_kwargs contains only Tensor values (caller already filtered
        # out non-tensor kwargs so they remain in the recompute_fn closure
        # without occupying GPU memory).
        self._num_positional = len(args)
        self._tensor_kwargs_keys = tuple(tensor_kwargs.keys())

        # Build one flat list: positional args first, then tensor kwargs values
        # in key insertion order.  This mirrors what _NoopSaveInputs will
        # receive, allowing get_inputs() to reconstruct both sides.
        all_flat = list(args) + list(tensor_kwargs.values())

        dummy = torch.empty((0,), requires_grad=True)
        hooks = _get_checkpoint_input_offload_hooks()
        if hooks is not None:
            pack_hook, unpack_hook = hooks
            for arg in all_flat:
                if isinstance(arg, torch.Tensor) and arg.is_cuda:
                    arg.activation_offloading = True
            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                self.input_saver = _NoopSaveInputs.apply(dummy, *all_flat)
        else:
            self.input_saver = _NoopSaveInputs.apply(dummy, *all_flat)

    def get_inputs(self):
        # Returns (pos_args, tensor_kwargs_dict) so that both positional and
        # keyword tensor inputs are retrieved via the offload/save mechanism
        # rather than from the recompute_fn closure.
        if self.input_saver is not None and self.input_saver.grad_fn is not None:
            flat = self.input_saver.grad_fn.get_args(self.input_saver.grad_fn.saved_tensors)
        else:
            flat = []

        pos_args = flat[:self._num_positional]
        kw_tensor_values = flat[self._num_positional:]
        tensor_kwargs = dict(zip(self._tensor_kwargs_keys, kw_tensor_values))
        return pos_args, tensor_kwargs

    def check_recomputed_tensors_match(self, gid) -> None:
        if self.ignore_saved_mismatch:
            return
        # NOTE [ Error handling for checkpoint ]
        #
        # At a high level, we need to check that the tensors saved
        # during original forward matches tensors saved during recompute.
        # This means handling 3 cases:
        #
        # 1. During recompute, more tensors were saved.
        if not len(self.weak_holders) == self.recomp_counter[gid]:
            # 2. During recompute, fewer tensors were saved
            raise CheckpointError(
                "torch.utils.checkpoint: A different number of tensors was saved "
                "during the original forward and recomputation.\n"
                f"Number of tensors saved during forward: {len(self.weak_holders)}\n"
                f"Number of tensors saved during recomputation: {self.recomp_counter[gid]}.\n"
                f"{_debug_tip_msg}"
            )

        # 3. During recompute, the same tensors were saved, but they
        #    have different metadata
        nb_meta_different = []
        for idx, weak_holder in enumerate(self.weak_holders):
            holder = weak_holder()
            if holder is None:
                continue
            _internal_assert(gid in holder.handles)
            _internal_assert(holder.handles[gid] is not None)
            _internal_assert(holder.handles[gid] in self.recomputed[gid])
            x_meta = self.x_metadatas[idx]
            recomputed_x = self.recomputed[gid][holder.handles[gid]]
            if x_meta != self.metadata_fn(recomputed_x):
                nb_meta_different.append((idx, x_meta, self.metadata_fn(recomputed_x)))

        if len(nb_meta_different) > 0:
            mismatched_tensors = ""
            for idx, x_meta, recomputed_meta in nb_meta_different:
                mismatched_tensors += (
                    f"tensor at position {idx}:\n"
                    f"saved metadata: {x_meta}\n"
                    f"recomputed metadata: {recomputed_meta}\n"
                )
            raise CheckpointError(
                "torch.utils.checkpoint: Recomputed values for the following tensors "
                "have different metadata than during the forward pass.\n"
                f"{mismatched_tensors}.\n"
                f"{_debug_tip_msg}"
            )


_debug_tip_msg = """
Tip: To see a more detailed error message, either pass `debug=True` to
`torch.utils.checkpoint.checkpoint(...)` or wrap the code block
with `with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):` to
enable checkpoint-debug mode globally.
"""


_checkpoint_error_template = """ \
An error happened while unpacking tensors; dumping logs of latest computation
because you passed `debug=True` to `torch.utils.checkpoint.checkpoint()`.
Scroll all the way down for guidance on how to navigate these logs.

+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|        1. Stack traces of the operators that ran in the original forward     |
+------------------------------------------------------------------------------+

{forward_traces}
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|        2. Stack traces of the operators that ran during recomputation        |
+------------------------------------------------------------------------------+

{recompute_traces}
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|       3. Log of operators in the original forward and recomputation          |
+------------------------------------------------------------------------------+
(Scroll up to correlate stack traces with each operation listed below. This
 helps identify their source in the code.)

IMPORTANT: Differences in "detach" calls between the original forward and the
           recomputation are expected. They are introduced by the checkpointing
           mechanism and can be ignored.

Operations executed during the original forward:

{forward_ops}

Operations executed during recomputation:

{recompute_ops}

+------------------------------------------------------------------------------+
 ERROR: Detected non-determinism while running activation checkpointing
--------------------------------------------------------------------------------
"""


class CheckpointError(RuntimeError):
    pass


def _get_debug_context_and_cb() -> Tuple[Callable[[], Any], Callable[[CheckpointError], None]]:
    # Import logging tensor lazily: torch.testing._internal is not part of
    # the stable public API and may be absent in some torch builds/versions.
    try:
        import platform as _platform
        from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
        _has_logging_tensor = True
    except ImportError:
        _has_logging_tensor = False

    if not _has_logging_tensor:
        # Degraded debug mode: no tensor-level traces, just re-raise the error.
        def unpack_error_cb_nolog(e: CheckpointError) -> NoReturn:
            raise e

        def context_fn_nolog():
            return contextlib.nullcontext(), contextlib.nullcontext()

        return context_fn_nolog, unpack_error_cb_nolog

    cpp_tb = _platform.machine() in ('x86_64', 'aarch64', 'arm64') and _platform.system() == 'Linux'

    class CaptureLogs:
        def __init__(self) -> None:
            self.logs = None
            self.tbs = None

        def get_context_manager(self):
            @contextlib.contextmanager
            def logging_mode():
                with LoggingTensorMode(), \
                     capture_logs(True, python_tb=True, script_tb=True, cpp_tb=cpp_tb) as logs_and_tb:
                    self.logs, self.tbs = logs_and_tb
                    yield logs_and_tb
            return logging_mode()

    capture_logs_fwd = CaptureLogs()
    capture_logs_recompute = CaptureLogs()

    def unpack_error_cb(e: CheckpointError) -> NoReturn:
        def get_str_tb(label, capture_logs_obj):
            out = ""
            total_len = len(capture_logs_obj.logs)
            for i, (log, tb) in enumerate(zip(capture_logs_obj.logs, capture_logs_obj.tbs)):
                out += f"{log}   ({i + 1} of {total_len} in {label})\n\n"
                found_torch_dispatch = False
                for line in tb:
                    is_torch_dispatch = line['name'] == '__torch_dispatch__'
                    if not found_torch_dispatch and not is_torch_dispatch:
                        continue
                    elif is_torch_dispatch:
                        found_torch_dispatch = True
                        continue
                    out += f"{line['filename']}:{line['line']}:{line['name']}\n"
                out += "\n\n"
            return out
        if capture_logs_fwd.logs is None:
            raise AssertionError("capture_logs_fwd.logs is None")
        if capture_logs_recompute.logs is None:
            raise AssertionError("capture_logs_recompute.logs is None")
        raise CheckpointError(
            _checkpoint_error_template.format(
                forward_traces=get_str_tb("original", capture_logs_fwd),
                recompute_traces=get_str_tb("recompute", capture_logs_recompute),
                forward_ops="\n".join(capture_logs_fwd.logs),
                recompute_ops="\n".join(capture_logs_recompute.logs)
            )
        ) from e

    def context_fn():
        return capture_logs_fwd.get_context_manager(), capture_logs_recompute.get_context_manager()

    return context_fn, unpack_error_cb


def _default_meta_extractor(x: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": x.shape,
        "dtype": x.dtype,
        "device": x.device
    }


_allowed_determinism_checks_to_fns: Dict[str, Callable[[torch.Tensor], Any]] = {
    _DEFAULT_DETERMINISM_MODE: _default_meta_extractor,
    "none": lambda _: None,
}


# See Rule 5
class _StopRecomputationError(Exception):
    pass


class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, target_frame_ref: ReferenceType, gid: GraphExecGroup | int) -> None:
        @torch._dynamo.disable
        def pack_hook(x):
            x = x.detach() if x.requires_grad else x
            target_frame = target_frame_ref()
            if target_frame is None:
                raise AssertionError("Internal error: target_frame reference is None")
            recomp_idx = target_frame.recomp_counter[gid]
            target_frame.recomp_counter[gid] += 1

            if recomp_idx >= len(target_frame.weak_holders):
                if target_frame.early_stop:
                    raise AssertionError("Unexpected state: target_frame.early_stop is set")
                if not target_frame.forward_completed:
                    target_frame.ignore_saved_mismatch = True
                    return x
                raise CheckpointError(
                    "torch.utils.checkpoint: trying to save more tensors during "
                    "recomputation than during the original forward pass.\n"
                    f"{_debug_tip_msg}"
                )

            holder = target_frame.weak_holders[recomp_idx]()

            if holder is not None:
                _internal_assert(holder.handles.get(gid, None) is None)
                holder.handles[gid] = _Handle()
                target_frame.recomputed[gid][holder.handles[gid]] = x

            if target_frame.early_stop and target_frame.recomp_counter[gid] == len(
                target_frame.weak_holders
            ):
                raise _StopRecomputationError
            return x

        def unpack_hook(x):
            return x

        super().__init__(pack_hook, unpack_hook)


# torch._disable_dynamo creates a reference cycle with decorated function.
# This function ensures no closure, so other objects aren't kept alive.
# https://github.com/pytorch/pytorch/issues/154642
@torch._disable_dynamo
def _run_fn_with_dynamo_disabled(fn, *args, **kwargs):
    return fn(*args, **kwargs)


class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame) -> None:
        def pack_hook(x):
            # See Rule 4 above
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            # Save metadata to detect non-determinism
            if frame.metadata_fn is not None:
                with torch.no_grad():
                    frame.x_metadatas.append(frame.metadata_fn(x))
            return holder

        def unpack_hook(holder):
            # First check if we're inside a GraphExecGroup context
            gid: GraphExecGroup | None | int = GraphExecGroup._get_current_group()
            if gid is None:
                gid = torch._C._current_graph_task_id()
                if gid == -1:
                    gid = int(uuid.uuid4())

            if not frame.is_recomputed[gid]:
                recomp_args, recomp_tensor_kwargs = frame.get_inputs()

                try:
                    with _recomputation_hook(
                        weakref.ref(frame), gid
                    ), torch.autograd.enable_grad():
                        # See Note: [compiled autograd and checkpoint unpack hook]
                        _run_fn_with_dynamo_disabled(
                            frame.recompute_fn, *recomp_args, **recomp_tensor_kwargs
                        )
                except _StopRecomputationError:
                    pass
                frame.is_recomputed[gid] = True
                frame.check_recomputed_tensors_match(gid)

            _internal_assert(gid in holder.handles)

            if holder.handles[gid] is None:
                extra = ""
                if _get_graph_exec_group() is not None:
                    extra = (
                        "Performing two backward calls that overlap (i.e. require the same "
                        "saved activation in order to compute gradients) is not allowed while "
                        "under the torch.utils.checkpoint.GraphExecGroup context. "
                    )
                raise CheckpointError(
                    "torch.utils.checkpoint: Unpack is being triggered for a tensor that was already "
                    f"unpacked once. {extra}If you are calling ctx.saved_tensors in backward, make sure "
                    "to do so only once. Otherwise please open an issue with details on your use case."
                )
            _internal_assert(holder.handles[gid] in frame.recomputed[gid])
            ret = frame.recomputed[gid][holder.handles[gid]]
            holder.handles[gid] = None
            return ret

        if frame.unpack_error_cb is not None:
            def unpack_hook_with_error_cb(holder):
                try:
                    return unpack_hook(holder)
                except CheckpointError as e:
                    frame.unpack_error_cb(e)
            super().__init__(pack_hook, unpack_hook_with_error_cb)
        else:
            super().__init__(pack_hook, unpack_hook)


def _is_compiling(func, args, kwargs):
    # Check if we are under AOTAutograd tracing
    return torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is not None


# NB: this helper wraps fn before calling checkpoint_impl. kwargs and
#     saving/restoring of global state is handled here.
def _checkpoint_without_reentrant_generator(
    fn,
    preserve_rng_state=True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    *args,
    **kwargs
):
    """Checkpointing without reentrant autograd.

    Args:
        fn: describes what to run in the forward pass of the model or
            part of the model.
        preserve_rng_state(bool, optional): Omit stashing and restoring
            the RNG state during each checkpoint. Default: ``True``
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. ``"default"`` or ``"none"``.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation.
        early_stop(bool, optional): If ``True``, stops recomputation as soon
            as it has computed all needed Tensors. Default: ``True``.
        *args: Arguments to pass in to the given ``function``.
        **kwargs: Keyword arguments to pass into the given ``function``.
    """
    unpack_error_cb = None

    if _checkpoint_debug_enabled if _checkpoint_debug_enabled is not None else debug:
        if context_fn is not noop_context_fn:
            raise ValueError(
                "debug=True is incompatible with non-default context_fn"
            )
        context_fn, unpack_error_cb = _get_debug_context_and_cb()

    if determinism_check in _allowed_determinism_checks_to_fns:
        metadata_fn = _allowed_determinism_checks_to_fns[determinism_check]
    else:
        raise ValueError(
            f"determinism_check should be one of {list(_allowed_determinism_checks_to_fns.keys())}, "
            f"but got {determinism_check}"
        )

    device_type = _infer_device_type(*args, *kwargs.values())
    device_module = _get_device_module(device_type)
    forward_context, recompute_context = context_fn()
    if _is_compiling(fn, args, kwargs) and context_fn is not noop_context_fn:
        if (
            not isinstance(forward_context, TorchDispatchMode)
            or not isinstance(recompute_context, TorchDispatchMode)
        ):
            raise AssertionError(
                "In torch.compile mode, `context_fn` arg passed to `torch.utils.checkpoint` "
                "must generate a tuple of two `TorchDispatchMode`s."
            )
    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device_type=device_type)

    if preserve_rng_state:
        fwd_cpu_state = torch.get_rng_state()
        had_device_in_fwd = False
        if getattr(device_module, "_initialized", False):
            had_device_in_fwd = True
            fwd_devices, fwd_device_states = get_device_states(*args, *kwargs.values())

    # recompute_fn should respect the device context of the original forward.
    # _get_current_function_mode_stack / DeviceContext are available only in
    # newer torch versions; fall back to a no-op context if not present.
    if _HAS_DEVICE_CTX:
        device_ctx = next(
            filter(
                lambda mode: isinstance(mode, DeviceContext),
                reversed(_get_current_function_mode_stack()),
            ),
            contextlib.nullcontext(),
        )
    else:
        device_ctx = contextlib.nullcontext()

    # Split kwargs into tensor-valued and non-tensor-valued parts.
    # Tensor-valued kwargs are saved via save_inputs() so the offload hooks
    # can manage their GPU memory.  They must NOT be captured by the
    # recompute_fn closure or they will remain alive on GPU until backward.
    # Non-tensor kwargs (None, bool, int, CPU tensors, …) are harmless to
    # keep in the closure and are merged back at recompute time.
    tensor_kwargs_for_save = {
        k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)
    }
    non_tensor_kwargs_closure = {
        k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)
    }

    def recompute_fn(*recomp_args, **recomp_tensor_kwargs) -> None:
        full_kwargs = {**non_tensor_kwargs_closure, **recomp_tensor_kwargs}
        rng_devices = []
        if preserve_rng_state and had_device_in_fwd:
            rng_devices = fwd_devices
        fork_rng_kwargs: Dict[str, Any] = {}
        if _fork_rng_supports_device_type:
            fork_rng_kwargs["device_type"] = device_type
        with torch.random.fork_rng(
            devices=rng_devices, enabled=preserve_rng_state, **fork_rng_kwargs
        ):
            if preserve_rng_state:
                torch.set_rng_state(fwd_cpu_state)
                if had_device_in_fwd:
                    set_device_states(fwd_devices, fwd_device_states, device_type=device_type)

            if device_autocast_kwargs is not None and _is_autocast_available(device_type):
                device_autocast_ctx = torch.amp.autocast(
                    device_type=device_type, **device_autocast_kwargs
                )
            else:
                device_autocast_ctx = contextlib.nullcontext()
            with device_autocast_ctx, torch.amp.autocast("cpu", **cpu_autocast_kwargs), recompute_context, device_ctx:  # type: ignore[attr-defined]
                _prev_recomputing = getattr(_recompute_state, "active", False)
                _recompute_state.active = True
                try:
                    fn(*recomp_args, **full_kwargs)
                finally:
                    _recompute_state.active = _prev_recomputing

    new_frame = _CheckpointFrame(
        recompute_fn,
        _enable_checkpoint_early_stop if _enable_checkpoint_early_stop is not None else early_stop,
        unpack_error_cb,
        metadata_fn
    )

    if not torch.is_grad_enabled():
        yield
        return

    new_frame.save_inputs(*args, **tensor_kwargs_for_save)

    with _checkpoint_hook(new_frame), forward_context:
        yield
    new_frame.forward_completed = True

    if getattr(device_module, "_initialized", False) and \
       preserve_rng_state and not had_device_in_fwd:  # type: ignore[possibly-undefined]
        raise RuntimeError(
            "PyTorch's device state was initialized in the forward pass "
            "of a Checkpoint, which is not allowed. Please open an issue "
            "if you need this feature."
        )

    return


class GraphExecGroup:
    """Any checkpointed regions encountered by backward under the same instance
    of this context manager will trigger recompute at most once, even if
    there are multiple calls to backward.

    Backward calls under the same instance of this context manager must execute
    over non-overlapping regions of the backward graph even if retain_graph=True.
    In particular, any two backward call cannot use the same saved activation for
    gradient computation.

    .. note::
        This context manager only affects checkpoint with use_reentrant=False.
    """

    def __enter__(self) -> Self:
        if _get_graph_exec_group() is not None:
            raise RuntimeError(
                "GraphExecGroup contexts cannot be nested. "
                f"Already inside group {_get_graph_exec_group()}"
            )
        _set_graph_exec_group(self)
        return self

    def __exit__(self, *args: object) -> None:
        _set_graph_exec_group(None)

    @classmethod
    def _get_current_group(cls) -> GraphExecGroup | None:
        return _get_graph_exec_group()


# Note: [compiled autograd and checkpoint unpack hook]
# When tracing via compiled autograd, this hook will be visible to the
# compiler if the forward of this checkpointed region ran in eager.
# If the forward had ran under compile, it would have been wrapped in a
# higher order op. See Note: [torch.compile and checkpoint].
#
# Since we run the recomputation hook under a enable_grad context,
# AOTDispatch will trace a joint graph for this hook, and may
# save different activations than in eager. This conflicts with the
# strict activation count checks in `frame.check_recomputed_tensors_match`.
# So, we disable this hook to force it to recompute eager checkpointed regions
# in eager. This could be removed if we can disable the partitioner for this
# hook.
