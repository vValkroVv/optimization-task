"""Focused review verification using CPU discovery stubs and a CUDA-init guard."""
import os
from pathlib import Path
import torch
import pytest


def reject_cuda(*args, **kwargs):
    raise AssertionError('CUDA discovery/initialization is forbidden in this review')


torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
for name in ('_lazy_init', 'current_device', 'get_device_properties', 'get_device_capability'):
    setattr(torch.cuda, name, reject_cuda)

if __name__ == '__main__':
    artifact_dir = Path(__file__).resolve().parent
    (artifact_dir / 'guarded-test-pid.txt').write_text(f'{os.getpid()}\n')
    raise SystemExit(pytest.main([
        '-p', 'no:cacheprovider', '-q',
        'tests/unit_tests/callbacks/test_campaign_cudagraph_step_boundary.py',
        'tests/unit_tests/test_config_torch_free.py::test_component_types_register_torch_free_and_are_pinned',
        'tests/unit_tests/test_campaign_continuation.py',
    ]))
