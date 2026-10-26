from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch

pytestmark = pytest.mark.unit

STREAM = 77


class _Compiled:
    launch_enter_hook = None
    launch_exit_hook = None
    function = "function"
    packed_metadata = "packed metadata"

    def __init__(self) -> None:
        self.runs: list[tuple] = []

    def run(self, *arguments) -> None:
        self.runs.append(arguments)

    def launch_metadata(self, grid, stream, *values):
        return ("metadata", grid, stream, values)


class _Kernel:
    """A JIT function: every `kernel[grid](...)` call is recorded, and returns the compiled kernel. An
    autotuned one has the config it tuned to."""

    def __init__(self, tuned: dict | None = None) -> None:
        self.calls: list[tuple] = []
        self.compiled = _Compiled()
        if tuned is not None:
            self.best_config = types.SimpleNamespace(kwargs=tuned)

    def __getitem__(self, grid):
        def call(*args, **constexprs):
            self.calls.append((grid, args, constexprs))
            return self.compiled

        return call


def _load(monkeypatch, version: str):
    triton = types.ModuleType("triton")
    triton.__version__ = version
    compiler = types.ModuleType("triton.compiler")
    compiler.CompiledKernel = _Compiled
    runtime = types.ModuleType("triton.runtime")
    runtime.driver = types.SimpleNamespace(
        active=types.SimpleNamespace(get_current_device=lambda: 0, get_current_stream=lambda device: STREAM)
    )
    triton.compiler, triton.runtime = compiler, runtime
    for name, module in (("triton", triton), ("triton.compiler", compiler), ("triton.runtime", runtime)):
        monkeypatch.setitem(sys.modules, name, module)
    # What the fresh import registers is removed again with the test.
    name = "metric_sampler.capture.kernels.triton_launch"
    monkeypatch.setitem(sys.modules, name, None)
    del sys.modules[name]
    monkeypatch.setattr(importlib.import_module("metric_sampler.capture.kernels"), "triton_launch", None, raising=False)
    return importlib.import_module(name)


@pytest.fixture
def triton_launch(monkeypatch):
    return _load(monkeypatch, "3.2.0")


def test_a_variants_first_launch_is_the_jit_call_and_later_launches_run_its_compiled_kernel(triton_launch):
    kernel = _Kernel()
    tensor = torch.zeros(64)
    args = (tensor, 5, 32, 1.5)

    def launch() -> None:
        triton_launch.launch(kernel, (4, 2, 1), *args, TILE=8, num_warps=4)

    launch()
    assert kernel.calls == [((4, 2, 1), args, {"TILE": 8, "num_warps": 4})]
    assert kernel.compiled.runs == []

    launch()
    launch()
    assert len(kernel.calls) == 1
    # Grid, stream, function, packed metadata, launch metadata, the two hooks, then the runtime arguments
    # with the tensor as its address.
    values = (tensor.data_ptr(), 5, 32, 1.5)
    assert kernel.compiled.runs == [
        (4, 2, 1, STREAM, "function", "packed metadata", ("metadata", (4, 2, 1), STREAM, values), None, None, *values)
    ] * 2


@pytest.mark.parametrize(
    ("change", "is_new_variant"),
    [
        ({"tile": 16}, True),
        ({"tensor": torch.zeros(64, dtype=torch.bfloat16)}, True),
        ({"tensor": torch.zeros(64)[1:]}, True),
        ({"integer": 32}, True),
        ({"integer": 1}, True),
        ({"integer": 7}, False),
    ],
    ids=["constexpr", "dtype", "address not a multiple of 16", "multiple of 16", "one", "same class of integer"],
)
def test_what_the_jit_specializes_on_makes_another_variant_with_its_own_launch(triton_launch, change, is_new_variant):
    kernel = _Kernel()
    base = {"tensor": torch.zeros(64), "integer": 5, "tile": 8}
    assert base["tensor"].data_ptr() % 16 == 0

    for case in (base, base | change):
        for _ in range(2):
            triton_launch.launch(kernel, (1, 1, 1), case["tensor"], case["integer"], TILE=case["tile"])

    assert len(kernel.calls) == 1 + is_new_variant
    assert len(kernel.compiled.runs) == 3 - is_new_variant


def test_variants_of_other_kernels_and_devices_are_kept_apart(triton_launch, monkeypatch):
    first, second = _Kernel(), _Kernel()
    tensor = torch.zeros(4)
    for kernel in (first, second):
        triton_launch.launch(kernel, (1, 1, 1), tensor)
    assert (len(first.calls), len(second.calls)) == (1, 1)

    monkeypatch.setattr(triton_launch.driver.active, "get_current_device", lambda: 1)
    triton_launch.launch(first, (1, 1, 1), tensor)
    assert len(first.calls) == 2


def test_an_autotuned_kernel_is_tuned_by_its_first_launch_and_later_launches_take_the_grid_of_what_it_tuned_to(
    triton_launch,
):
    kernel = _Kernel(tuned={"BLOCK_M": 64})

    def grid(meta: dict[str, int]) -> tuple[int, int, int]:
        return -(-100 // meta["BLOCK_M"]), 2, 3
    tensor = torch.zeros(8)

    for _ in range(3):
        triton_launch.launch_autotuned(kernel, grid, tensor, 5, TILE=8)

    assert kernel.calls == [(grid, (tensor, 5), {"TILE": 8})]
    values = (tensor.data_ptr(), 5)
    assert kernel.compiled.runs == [
        (2, 2, 3, STREAM, "function", "packed metadata", ("metadata", (2, 2, 3), STREAM, values), None, None, *values)
    ] * 2

    triton_launch.launch_autotuned(kernel, grid, tensor, 32, TILE=8)

    assert len(kernel.calls) == 2


def test_launches_in_a_jit_block_are_the_jit_call_and_the_direct_launch_resumes_after_it(triton_launch):
    kernel = _Kernel()
    tensor = torch.zeros(4)
    triton_launch.launch(kernel, (1, 1, 1), tensor)

    with triton_launch.jit_launches():
        triton_launch.launch(kernel, (1, 1, 1), tensor)
        assert len(kernel.calls) == 2

    triton_launch.launch(kernel, (1, 1, 1), tensor)
    assert len(kernel.calls) == 2 and len(kernel.compiled.runs) == 1


def test_a_failure_in_a_jit_block_leaves_the_direct_launch_on(triton_launch):
    with pytest.raises(ZeroDivisionError), triton_launch.jit_launches():
        raise ZeroDivisionError

    kernel = _Kernel()
    for _ in range(2):
        triton_launch.launch(kernel, (1, 1, 1), torch.zeros(4))
    assert len(kernel.calls) == 1


@pytest.mark.parametrize("version", ["3.1.0", "3.3.1", "2.9.0"])
def test_other_triton_versions_keep_the_jit_call_for_every_launch(monkeypatch, version):
    triton_launch = _load(monkeypatch, version)
    kernel, autotuned = _Kernel(), _Kernel(tuned={"BLOCK_M": 64})
    for _ in range(3):
        triton_launch.launch(kernel, (2, 1, 1), torch.zeros(4), 3, TILE=8)
        triton_launch.launch_autotuned(autotuned, lambda meta: (2, 1, 1), torch.zeros(4), 3, TILE=8)
    assert len(kernel.calls) == len(autotuned.calls) == 3
    assert kernel.compiled.runs == autotuned.compiled.runs == []
