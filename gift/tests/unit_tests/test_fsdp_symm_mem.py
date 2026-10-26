"""The symm-mem hardware check lives at the point of use, not in config."""

import pytest
import torch
import torch.nn as nn

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.fsdp import enable_fsdp_symm_mem


def test_parallelism_config_does_not_probe_hardware(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ParallelismConfig(enable_fsdp_symm_mem=True).enable_fsdp_symm_mem


def test_enable_fsdp_symm_mem_rejects_unsupported_hardware(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="compute capability 9.0"):
        enable_fsdp_symm_mem(nn.Linear(2, 2))
