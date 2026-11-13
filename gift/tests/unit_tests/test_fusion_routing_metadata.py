"""Focused correctness checks for explicit HybridEP routing metadata."""

import pytest
import torch

from torchtitan.distributed.deepep.routing import (
    gather_route_gradients,
    indices_to_map as reference_indices_to_map,
)
from torchtitan.experiments.four_fusion.routing_metadata import (
    _routing_metadata_cuda,
    _can_use_cuda_kernel,
    indices_to_map as triton_indices_to_map,
)


_SMALL_IDS = torch.tensor(
    [[0, -1, 2], [1, 3, -1], [-1, -1, -1], [2, 1, 0]], dtype=torch.int64
)
_SMALL_WEIGHTS = torch.tensor(
    [
        [0.0, float("nan"), 0.25],
        [0.5, 0.75, float("nan")],
        [float("nan"), float("nan"), float("nan")],
        [0.125, 0.25, 0.875],
    ],
    dtype=torch.float32,
)
_EXPECTED_ROUTING = torch.tensor(
    [
        [True, False, True, False],
        [False, True, False, True],
        [False, False, False, False],
        [True, True, True, False],
    ],
    dtype=torch.bool,
)
_EXPECTED_PROBS = torch.tensor(
    [
        [0.0, 0.0, 0.25, 0.0],
        [0.0, 0.5, 0.0, 0.75],
        [0.0, 0.0, 0.0, 0.0],
        [0.875, 0.25, 0.125, 0.0],
    ],
    dtype=torch.float32,
)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="requires an available CUDA device",
            ),
        ),
    ]
)
def device(request):
    return torch.device(request.param)


def _noncontiguous_inputs(device, id_dtype, weight_dtype, *, requires_grad=False):
    ids_storage = torch.full((4, 6), -1, dtype=id_dtype, device=device)
    ids = ids_storage[:, ::2]
    ids.copy_(_SMALL_IDS.to(device=device, dtype=id_dtype))

    weights_storage = torch.empty((4, 6), dtype=weight_dtype, device=device)
    weights = weights_storage[:, ::2]
    weights.copy_(_SMALL_WEIGHTS.to(device=device, dtype=weight_dtype))
    if requires_grad:
        weights_storage.requires_grad_()
        weights = weights_storage[:, ::2]
    return ids, weights, weights_storage


def _assert_one_sink_storage(routing, probabilities, num_tokens, num_experts):
    num_entries = num_tokens * num_experts
    assert routing.shape == (num_tokens, num_experts)
    assert probabilities.shape == (num_tokens, num_experts)
    assert routing.dtype == torch.bool
    assert probabilities.dtype == torch.float32
    assert routing.is_contiguous() and probabilities.is_contiguous()
    assert routing.untyped_storage().nbytes() == num_entries + 1
    assert probabilities.untyped_storage().nbytes() == (num_entries + 1) * 4


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_metadata_matches_reference_for_signed_strided_inputs(
    device, id_dtype, weight_dtype
):
    ids, weights, _ = _noncontiguous_inputs(device, id_dtype, weight_dtype)

    routing, probabilities = triton_indices_to_map(ids, weights, 4)
    reference_routing, reference_probabilities = reference_indices_to_map(
        ids, weights, 4
    )

    torch.testing.assert_close(routing, _EXPECTED_ROUTING.to(device), rtol=0, atol=0)
    torch.testing.assert_close(
        probabilities, _EXPECTED_PROBS.to(device), rtol=0, atol=0
    )
    torch.testing.assert_close(routing, reference_routing, rtol=0, atol=0)
    torch.testing.assert_close(probabilities, reference_probabilities, rtol=0, atol=0)
    _assert_one_sink_storage(routing, probabilities, 4, 4)
    assert routing.untyped_storage().data_ptr() != ids.untyped_storage().data_ptr()
    assert (
        probabilities.untyped_storage().data_ptr()
        != weights.untyped_storage().data_ptr()
    )
    assert (
        routing.untyped_storage().data_ptr()
        != probabilities.untyped_storage().data_ptr()
    )
    assert routing[0, 0] and probabilities[0, 0] == 0


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_metadata_preserves_standalone_weight_gradients_and_base_view_gradients(
    device, weight_dtype
):
    ids, weights, weights_storage = _noncontiguous_inputs(
        device, torch.int64, weight_dtype, requires_grad=True
    )
    reference_storage = weights_storage.detach().clone().requires_grad_()
    reference_weights = reference_storage[:, ::2]
    dense_grad = torch.arange(1, 17, dtype=torch.float32, device=device).view(4, 4)

    _, probabilities = triton_indices_to_map(ids, weights, 4)
    _, reference_probabilities = reference_indices_to_map(ids, reference_weights, 4)
    (probabilities * dense_grad).sum().backward()
    (reference_probabilities * dense_grad).sum().backward()

    torch.testing.assert_close(
        weights_storage.grad, reference_storage.grad, rtol=0, atol=0
    )
    expected = gather_route_gradients(dense_grad, ids).to(weight_dtype)
    torch.testing.assert_close(weights_storage.grad[:, ::2], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        weights_storage.grad[:, 1::2],
        torch.zeros_like(weights_storage.grad[:, 1::2]),
        rtol=0,
        atol=0,
    )


def test_metadata_preserves_signed_zero_and_valid_nan_bits(device):
    ids = torch.tensor([[0, 1, -1]], dtype=torch.int64, device=device)
    weights = torch.tensor(
        [[-0.0, float("nan"), float("nan")]], dtype=torch.float32, device=device
    )

    routing, probabilities = triton_indices_to_map(ids, weights, 4)
    reference_routing, reference_probabilities = reference_indices_to_map(
        ids, weights, 4
    )

    torch.testing.assert_close(routing, reference_routing, rtol=0, atol=0)
    assert torch.equal(
        probabilities.view(torch.int32), reference_probabilities.view(torch.int32)
    )
    assert routing[0, 0] and routing[0, 1]
    assert probabilities[0, 0].view(torch.int32).item() == -2147483648
    assert torch.isnan(probabilities[0, 1])
    assert torch.equal(
        probabilities[0, 2:].view(torch.int32),
        torch.zeros(2, dtype=torch.int32, device=device),
    )


def test_metadata_accepts_empty_token_batch(device):
    ids = torch.empty((0, 3), dtype=torch.int64, device=device)
    weights = torch.empty((0, 3), dtype=torch.float32, device=device)

    routing, probabilities = triton_indices_to_map(ids, weights, 128)
    reference_routing, reference_probabilities = reference_indices_to_map(
        ids, weights, 128
    )

    _assert_one_sink_storage(routing, probabilities, 0, 128)
    torch.testing.assert_close(routing, reference_routing, rtol=0, atol=0)
    torch.testing.assert_close(probabilities, reference_probabilities, rtol=0, atol=0)


def test_metadata_production_shape_t8192_e128_k8(device):
    if device.type != "cuda":
        pytest.skip("the explicit production-shape kernel check requires CUDA")
    num_tokens, num_experts, top_k = 8192, 128, 8
    rows = torch.arange(num_tokens, dtype=torch.int64, device=device)[:, None]
    slots = torch.arange(top_k, dtype=torch.int64, device=device)[None, :]
    ids_storage = torch.empty((num_tokens, top_k * 2), dtype=torch.int64, device=device)
    ids = ids_storage[:, ::2]
    ids.copy_((rows + slots) % num_experts)
    ids[:, -1] = -1
    weights_storage = torch.ones(
        (num_tokens, top_k * 2), dtype=torch.float32, device=device
    )
    weights = weights_storage[:, ::2]
    weights[:, 0] = 0
    weights[:, -1] = float("nan")

    routing, probabilities = triton_indices_to_map(ids, weights, num_experts)
    reference_routing, reference_probabilities = reference_indices_to_map(
        ids, weights, num_experts
    )

    torch.testing.assert_close(routing, reference_routing, rtol=0, atol=0)
    torch.testing.assert_close(probabilities, reference_probabilities, rtol=0, atol=0)
    _assert_one_sink_storage(routing, probabilities, num_tokens, num_experts)
    assert routing.sum().item() == num_tokens * (top_k - 1)
    assert torch.isfinite(probabilities).all().item()
    assert probabilities.sum().item() == num_tokens * (top_k - 2)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an available CUDA device"
)
def test_opcheck_and_outer_compile_forward_backward():
    device = torch.device("cuda")
    ids, weights, weights_storage = _noncontiguous_inputs(
        device, torch.int32, torch.float32, requires_grad=True
    )
    forward_opcheck = torch.library.opcheck(
        _routing_metadata_cuda, (ids, weights.detach(), 4)
    )
    assert set(forward_opcheck.values()) == {"SUCCESS"}, forward_opcheck
    grad_opcheck = torch.library.opcheck(
        _routing_metadata_cuda,
        (ids, weights, 4),
        test_utils=(
            "test_schema",
            "test_autograd_registration",
            "test_faketensor",
        ),
    )
    assert set(grad_opcheck.values()) == {"SUCCESS"}, grad_opcheck

    dense_grad = torch.arange(1, 17, dtype=torch.float32, device=device).view(4, 4)

    def route(ids_arg, weights_arg):
        routing, probabilities = triton_indices_to_map(ids_arg, weights_arg, 4)
        return routing, (probabilities * dense_grad).sum()

    compiled_route = torch.compile(route, fullgraph=True)
    routing, loss = compiled_route(ids, weights)
    loss.backward()
    assert torch.equal(routing, _EXPECTED_ROUTING.to(device))
    assert weights_storage.grad is not None
    expected_grad = gather_route_gradients(dense_grad, ids).to(weights.dtype)
    torch.testing.assert_close(
        weights_storage.grad[:, ::2], expected_grad, rtol=0, atol=0
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an available CUDA device"
)
def test_large_expert_vector_uses_eager_reference_fallback():
    device = torch.device("cuda")
    ids = torch.tensor([[2, -1], [1024, 1]], dtype=torch.int64, device=device)
    weights = torch.tensor([[0.5, float("nan")], [0.75, 0.25]], device=device)
    assert not _can_use_cuda_kernel(ids, weights, 1025)
    actual = triton_indices_to_map(ids, weights, 1025)
    expected = reference_indices_to_map(ids, weights, 1025)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an available CUDA device"
)
def test_triton_metadata_runs_inside_hybridep_dispatch_custom_op(monkeypatch):
    from torchtitan.distributed.deepep import hybridep

    if hybridep.indices_to_map is not triton_indices_to_map:
        pytest.skip("requires the candidate routing-metadata import patch")

    device = torch.device("cuda")
    ids, weights, _ = _noncontiguous_inputs(device, torch.int64, torch.float32)
    hidden_in = torch.ones((4, 8), dtype=torch.bfloat16, device=device)
    expected_routing = _EXPECTED_ROUTING.to(device)
    expected_probabilities = _EXPECTED_PROBS.to(device)

    class RecordingBuffer:
        def dispatch_with_permute(self, **kwargs):
            torch.testing.assert_close(
                kwargs["routing_map"], expected_routing, rtol=0, atol=0
            )
            torch.testing.assert_close(
                kwargs["probs"], expected_probabilities, rtol=0, atol=0
            )
            return (
                kwargs["hidden"].clone(),
                kwargs["probs"],
                torch.empty(0, dtype=torch.float32, device=device),
                torch.ones(4, dtype=torch.int32, device="cpu"),
                object(),
            )

    buffer = RecordingBuffer()
    monkeypatch.setattr(
        hybridep.dist.distributed_c10d,
        "_resolve_process_group",
        lambda _: object(),
    )
    monkeypatch.setattr(
        hybridep,
        "get_buffer",
        lambda **_: setattr(hybridep, "_buffer", buffer),
    )
    monkeypatch.setattr(hybridep, "_COMM_STREAM_ENABLED", False)
    monkeypatch.setattr(hybridep, "_FUSE_PERMUTE", False)

    result = torch.ops.hybridep.dispatch.default(
        hidden_in,
        ids,
        weights,
        4,
        1,
        "routing_metadata_test",
        [],
        False,
        None,
        None,
        False,
    )

    assert result[2].shape == (4, 4)
    torch.testing.assert_close(result[2], expected_probabilities, rtol=0, atol=0)
