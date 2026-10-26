from __future__ import annotations

from typing import SupportsInt

import numpy as np
import torch

_CUDA_HOST_REGISTER_DEFAULT = 0


def _check_cudart(error: SupportsInt, call: str) -> None:
    if int(error) != 0:
        raise RuntimeError(f"{call} failed with CUDA error {int(error)}")


def register_host_memory(region: np.ndarray) -> int:
    """Page-locks `region` so that copies into it can run asynchronously; returns its address."""
    address = region.ctypes.data
    _check_cudart(
        torch.cuda.cudart().cudaHostRegister(address, region.nbytes, _CUDA_HOST_REGISTER_DEFAULT), "cudaHostRegister"
    )
    return address


def unregister_host_memory(address: int) -> None:
    _check_cudart(torch.cuda.cudart().cudaHostUnregister(address), "cudaHostUnregister")
