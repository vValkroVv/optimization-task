"""Exact Triton mask checks on CUDA, or with TRITON_INTERPRET=1 on CPU."""

import os

import pytest
import torch

from torchtitan.models.common.moe.dispatch.capacity import _branchless_capacity_mask

pytest.importorskip("triton")
from torchtitan.ops.token_drop import capacity_mask  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and not INTERPRET,
    reason="requires CUDA or the explicitly enabled Triton interpreter",
)
DEVICE = "cpu" if INTERPRET else "cuda"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_routes", [0, 1, 17, 257, 1025])
@pytest.mark.parametrize("capacity", [0, 1, 13, 2048])
@pytest.mark.parametrize("needs_drop", [False, True])
def test_radix_mask_matches_stable_sort(dtype, num_routes, capacity, needs_drop):
    torch.manual_seed(42)
    num_experts = 7
    # Quantization deliberately introduces ties and signed/zero scores.
    weights = (torch.randn(num_routes, device=DEVICE) * 4).round().to(dtype)
    buckets = torch.randint(0, num_experts, (num_routes,), device=DEVICE)
    valid = (weights != 0) & (torch.arange(num_routes, device=DEVICE) % 11 != 0)
    buckets = torch.where(valid, buckets, num_experts)
    counts = torch.bincount(buckets, minlength=num_experts + 1)
    predicate = torch.tensor(needs_drop, device=DEVICE)
    actual = capacity_mask(weights, buckets, valid, counts, predicate, capacity)
    expected = _branchless_capacity_mask(weights, buckets, valid, counts, capacity)
    expected = torch.where(predicate, expected, valid)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_routes", [4097, 32768, 65536])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_large_skewed_routes_and_empty_experts(num_routes, dtype):
    torch.manual_seed(17)
    weights = torch.randn(num_routes, device=DEVICE).to(dtype)
    # Two active experts, many empty ones; both selection and copy branches run.
    buckets = torch.zeros(num_routes, dtype=torch.long, device=DEVICE)
    buckets[-5:] = 31
    valid = weights != 0
    buckets = torch.where(valid, buckets, 32)
    counts = torch.bincount(buckets, minlength=33)
    capacity = num_routes // 8
    actual = capacity_mask(
        weights, buckets, valid, counts, torch.tensor(True, device=DEVICE), capacity
    )
    expected = _branchless_capacity_mask(weights, buckets, valid, counts, capacity)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_equal_scores_keep_original_route_order():
    weights = torch.ones(513, device=DEVICE)
    buckets = torch.arange(513, device=DEVICE) % 3
    valid = torch.ones(513, dtype=torch.bool, device=DEVICE)
    counts = torch.bincount(buckets, minlength=4)
    actual = capacity_mask(
        weights, buckets, valid, counts, torch.tensor(True, device=DEVICE), 2
    )
    expected = torch.arange(513, device=DEVICE) < 6
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_strided_scores_and_extreme_magnitudes(dtype):
    tiny = torch.finfo(dtype).tiny
    largest = torch.finfo(dtype).max
    values = torch.tensor(
        [
            tiny / 8,
            tiny,
            -tiny,
            -tiny / 8,
            largest,
            -largest,
            float("inf"),
            -float("inf"),
            1,
            1,
        ],
        device=DEVICE,
        dtype=dtype,
    )
    storage = torch.zeros(values.numel() * 2, device=DEVICE, dtype=dtype)
    storage[::2] = values
    weights = storage[::2]
    buckets = torch.zeros(weights.numel(), dtype=torch.long, device=DEVICE)
    valid = torch.ones_like(weights, dtype=torch.bool)
    counts = torch.tensor([weights.numel(), 0], device=DEVICE)
    actual = capacity_mask(
        weights, buckets, valid, counts, torch.tensor(True, device=DEVICE), 6
    )
    expected = _branchless_capacity_mask(weights, buckets, valid, counts, 6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    INTERPRET, reason="production CUDA dispatch selection is checked on a real GPU"
)
def test_selector_launches_triton_without_reading_device_scalars(monkeypatch):
    from unittest.mock import Mock
    from torch.utils._python_dispatch import TorchDispatchMode
    from torchtitan.config.moe import TokenDropConfig
    from torchtitan.models.common.moe.dispatch.capacity import select_capacity_routes
    import torchtitan.ops.token_drop as native

    class NoScalarReads(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                raise AssertionError("Branchless selection must not read a GPU scalar")
            return func(*args, **(kwargs or {}))

    scores = torch.linspace(0.1, 1.0, 514, device=DEVICE).reshape(257, 2)
    ids = torch.arange(514, device=DEVICE).reshape(257, 2) % 7
    policy = TokenDropConfig(
        expert_capacity_factor=0.5,
        drop_by_group_capacity=True,
        branchless_capacity_drop=True,
    )
    launch = Mock(wraps=native.capacity_mask)
    monkeypatch.setattr(native, "capacity_mask", launch)
    with NoScalarReads():
        actual = select_capacity_routes(scores, ids, num_experts=7, policy=policy)
    launch.assert_called_once()
    weights = scores.reshape(-1)
    buckets = ids.reshape(-1)
    valid = weights != 0
    counts = torch.bincount(buckets, minlength=8)
    capacity = policy.source_capacity(num_tokens=257, top_k=2, num_experts=7, ep_size=1)
    expected = _branchless_capacity_mask(weights, buckets, valid, counts, capacity)
    torch.testing.assert_close(actual.reshape(-1), expected, rtol=0, atol=0)


@pytest.mark.skipif(INTERPRET, reason="CUDA fallback dispatch is checked on a real GPU")
@pytest.mark.parametrize(
    "dtype,num_tokens", [(torch.float64, 1031), (torch.float32, 65537)]
)
def test_cuda_fallback_preserves_unsupported_shapes_and_dtypes(
    monkeypatch, dtype, num_tokens
):
    from unittest.mock import Mock
    from torchtitan.config.moe import TokenDropConfig
    from torchtitan.models.common.moe.dispatch.capacity import select_capacity_routes
    import torchtitan.ops.token_drop as native

    launch = Mock(
        side_effect=AssertionError("Unsupported inputs must use the PyTorch fallback")
    )
    monkeypatch.setattr(native, "capacity_mask", launch)
    scores = torch.arange(num_tokens, 0, -1, device=DEVICE, dtype=dtype).reshape(-1, 1)
    ids = torch.zeros_like(scores, dtype=torch.long)
    policy = TokenDropConfig(
        strategy="fixed_capacity",
        expert_capacity_tokens=17,
        branchless_capacity_drop=True,
    )
    actual = select_capacity_routes(scores, ids, num_experts=2, policy=policy)
    expected = (torch.arange(num_tokens, device=DEVICE) < 17).reshape(-1, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    launch.assert_not_called()
