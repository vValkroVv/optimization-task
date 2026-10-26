from __future__ import annotations

import numpy as np
import pytest
import torch

from metric_sampler.capture import host_registration

pytestmark = pytest.mark.unit

_FAILURE_CODE = 700


class _FakeCudaRuntime:
    def __init__(self, error: int) -> None:
        self.error = error
        self.calls: list[tuple] = []

    def cudaHostRegister(self, address: int, size: int, flags: int) -> int:
        self.calls.append(("cudaHostRegister", address, size, flags))
        return self.error

    def cudaHostUnregister(self, address: int) -> int:
        self.calls.append(("cudaHostUnregister", address))
        return self.error


@pytest.fixture
def runtime(monkeypatch) -> _FakeCudaRuntime:
    fake = _FakeCudaRuntime(error=0)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: fake)
    return fake


def test_register_host_memory_registers_the_region_with_the_default_flags_and_returns_its_address(runtime) -> None:
    region = np.zeros(128, dtype=np.uint8)

    address = host_registration.register_host_memory(region)

    assert address == region.ctypes.data
    assert runtime.calls == [("cudaHostRegister", address, 128, 0)]


def test_unregister_host_memory_unregisters_the_address(runtime) -> None:
    host_registration.unregister_host_memory(4096)

    assert runtime.calls == [("cudaHostUnregister", 4096)]


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (lambda: host_registration.register_host_memory(np.zeros(8, dtype=np.uint8)), "cudaHostRegister"),
        (lambda: host_registration.unregister_host_memory(4096), "cudaHostUnregister"),
    ],
)
def test_a_cuda_error_fails_naming_the_call_and_the_code(runtime, call, name) -> None:
    runtime.error = _FAILURE_CODE

    with pytest.raises(RuntimeError, match=f"{name} failed with CUDA error {_FAILURE_CODE}"):
        call()
