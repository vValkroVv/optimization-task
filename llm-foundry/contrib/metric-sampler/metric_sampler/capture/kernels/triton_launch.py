from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import torch
import triton
from triton.compiler import CompiledKernel
from triton.runtime import driver

from metric_sampler.capture.kernels import activation_ext

if TYPE_CHECKING:
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

# `kernel[grid](...)` binds and specializes every argument in Python on each call. `launch` pays that once per
# variant and later calls the kernel it compiled directly, through the prebuilt extension when there is one, which
# is the kernel the JIT call picks. The direct call follows `JITFunction.run` and `CompiledKernel.run` of Triton 3.2;
# with any other version every launch is the JIT call.
_DIRECT_LAUNCH = triton.__version__.startswith("3.2.")

# A launched variant: its compiled kernel, for an autotuned kernel the meta-parameters it was tuned to (empty for
# another), and the extension's launch of the compiled kernel when there is one.
_Launched = tuple[CompiledKernel, dict[str, int], Callable[..., None] | None]
# Variants are told apart by what the JIT specializes on: the constexpr arguments, a tensor's dtype and whether its
# address is a multiple of 16, an integer's divisibility by 16 and equality to 1.
_compiled: dict[tuple[object, ...], _Launched] = {}

_DIVISIBILITY = 16
# The classes the JIT specializes an integer on, numbered as the extension's `triton_arguments` numbers them.
_DIVISIBLE = 0
_EQUAL_TO_ONE = 1
_OTHER = 2
# The kinds of runtime argument other than pointers that the extension's launch takes, as Triton names them.
_NATIVE_KINDS = frozenset({"i1", "i32", "i64", "u64", "fp32"})


def launch(
    kernel: JITFunction, grid: tuple[int, int, int], *args: torch.Tensor | int | float, **constexprs: int,
) -> None:
    """`kernel[grid](*args, **constexprs)`, for `args` the kernel's runtime arguments: tensors, integers that
    fit int32 and floats. Constexprs are always passed in the same order."""
    if not _DIRECT_LAUNCH:
        kernel[grid](*args, **constexprs)
        return
    device, key, values = _variant(kernel, args, constexprs)
    launched = _compiled.get(key)
    if launched is None:
        compiled = kernel[grid](*args, **constexprs)
        _compiled[key] = compiled, {}, _native(compiled)
        return
    _run(launched, device, grid, values)


def launch_autotuned(
    kernel: Autotuner, grid: Callable[[dict[str, int]], tuple[int, int, int]], *args: torch.Tensor | int | float,
    **constexprs: int,
) -> None:
    """`launch` of an autotuned kernel, `grid` a function of its tuned meta-parameters as for the JIT call.
    A variant is tuned by its first launch, which is the JIT call's."""
    if not _DIRECT_LAUNCH:
        kernel[grid](*args, **constexprs)
        return
    device, key, values = _variant(kernel, args, constexprs)
    launched = _compiled.get(key)
    if launched is None:
        compiled = kernel[grid](*args, **constexprs)
        _compiled[key] = compiled, kernel.best_config.kwargs, _native(compiled)
        return
    _, tuned_parameters, _ = launched
    _run(launched, device, grid(tuned_parameters), values)


@contextlib.contextmanager
def jit_launches() -> Iterator[None]:
    """The JIT call for every launch in the block, for checks that compare direct launches with it."""
    global _DIRECT_LAUNCH
    previous, _DIRECT_LAUNCH = _DIRECT_LAUNCH, False
    try:
        yield
    finally:
        _DIRECT_LAUNCH = previous


def _variant(
    kernel: JITFunction | Autotuner, args: tuple[torch.Tensor | int | float, ...], constexprs: dict[str, int],
) -> tuple[int, tuple[object, ...], list[int | float]]:
    """The current device, the key of the variant the JIT would launch, and `args` as the launcher takes them."""
    device = driver.active.get_current_device()
    extension = activation_ext.prebuilt()
    if extension:  # pragma: no cover
        # The same classes and values from one C++ call, without a Python step per argument.
        specialization, launcher_values = extension.triton_arguments(args)
        return device, (kernel, device, *constexprs.values(), *specialization), launcher_values
    variant: list[object] = [kernel, device, *constexprs.values()]
    values: list[int | float] = []
    for arg in args:
        value: int | float
        if isinstance(arg, torch.Tensor):
            value = arg.data_ptr()
            variant += (arg.dtype, value % _DIVISIBILITY == 0)
        else:
            value = arg
            if value % _DIVISIBILITY == 0:
                variant.append(_DIVISIBLE)
            elif value == 1:
                variant.append(_EQUAL_TO_ONE)
            else:
                variant.append(_OTHER)
        values.append(value)
    return device, tuple(variant), values


def _native(compiled: CompiledKernel) -> Callable[..., None] | None:
    """The extension's launch of `compiled`, or None without the extension or for a kernel it does not take. Its
    arguments are those of Triton 3.2's launcher: the kernel's signature, in order, but its constants, which
    CudaLauncher keys by argument name or index alike."""
    extension = activation_ext.prebuilt()
    if extension:  # pragma: no cover
        source = compiled.src

        def argument_index(name: str | int) -> int:
            return source.fn.arg_names.index(name) if isinstance(name, str) else name

        constants = {argument_index(name) for name in getattr(source, "constants", {})}
        kinds = ["" if argument_index(name) in constants else kind for name, kind in source.signature.items()]
        if compiled.metadata.num_ctas == 1 and all(
                not kind or kind.startswith("*") or kind in _NATIVE_KINDS for kind in kinds):
            return extension.TritonKernel(
                compiled.function, compiled.metadata.num_warps, compiled.metadata.shared, kinds)
    return None


def _run(launched: _Launched, device: int, grid: tuple[int, int, int], values: list[int | float]) -> None:
    compiled, _, native = launched
    # The launch hooks, which a profiler sets, go through the launcher.
    if (native is not None and CompiledKernel.launch_enter_hook is None
            and CompiledKernel.launch_exit_hook is None):  # pragma: no cover
        native(*grid, values)
        return
    stream = driver.active.get_current_stream(device)
    compiled.run(
        *grid, stream, compiled.function, compiled.packed_metadata, compiled.launch_metadata(grid, stream, *values),
        CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, *values,
    )
