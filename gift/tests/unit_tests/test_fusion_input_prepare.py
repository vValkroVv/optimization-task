from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from torchtitan.components.aux_loss import AuxLossTracker
from torchtitan.components.loss import IGNORE_INDEX, MTPLoss
from torchtitan.models.common.attention import AttentionMaskKind, MaskType
from torchtitan.models.common.decoder_blocks import Decoder
from torchtitan.models.common.regularization import (
    AddAuxiliaryLoss,
    aux_loss_args,
    declare_aux_loss,
)
from torchtitan.trainer import Trainer
from torchtitan.experiments.four_fusion import input_prepare
from torchtitan.experiments.four_fusion.input_prepare import (
    MAX_PREPARE_ELEMENTS,
    _prepare_input_tensors_cuda,
    _prepare_input_tensors_cuda_scalar,
    _prepare_tensors,
    prepare_forward_batch,
)


def _cuda_available():
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable"
    )


def _reference(
    labels: torch.Tensor,
    supplied_positions: torch.Tensor | None,
    *,
    generate_positions: bool,
    pack_mtp_targets: bool,
):
    if generate_positions:
        positions = torch.arange(
            labels.shape[1], dtype=torch.int32, device=labels.device
        ).repeat(labels.shape[0], 1)
    else:
        positions = supplied_positions
    token_mask = labels != IGNORE_INDEX
    valid_tokens = token_mask.sum(dtype=torch.int64)
    if pack_mtp_targets and positions is not None:
        packed_targets = torch.stack((labels, positions.to(dtype=labels.dtype)), dim=1)
    else:
        packed_targets = labels
    return positions, token_mask, valid_tokens, packed_targets


def _assert_preparation_equal(actual, expected):
    actual_positions, actual_mask, actual_count, actual_targets = actual[:4]
    expected_positions, expected_mask, expected_count, expected_targets = expected
    if expected_positions is None:
        assert actual_positions is None
    else:
        torch.testing.assert_close(actual_positions, expected_positions, rtol=0, atol=0)
    torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)
    assert actual_mask.dtype is torch.bool
    assert actual_mask.is_contiguous()
    torch.testing.assert_close(actual_count, expected_count, rtol=0, atol=0)
    assert actual_count.dtype is torch.int64
    torch.testing.assert_close(actual_targets, expected_targets, rtol=0, atol=0)


@_cuda_available()
def test_explicit_kernel_generates_positions_masks_counts_and_packs_strided_labels():
    device = torch.device("cuda:0")
    storage = torch.tensor(
        [
            [11, 70, IGNORE_INDEX, 71, 22, 72, 33, 73],
            [IGNORE_INDEX, 80, 44, 81, IGNORE_INDEX, 82, 55, 83],
        ],
        dtype=torch.int64,
        device=device,
    )
    labels = storage[:, ::2]
    before = storage.clone()
    assert not labels.is_contiguous()
    global_valid_tokens = torch.tensor(100, dtype=torch.int64, device=device)
    actual = _prepare_tensors(
        labels,
        None,
        generate_positions=True,
        pack_mtp_targets=True,
        global_valid_tokens=global_valid_tokens,
    )
    expected = _reference(labels, None, generate_positions=True, pack_mtp_targets=True)

    _assert_preparation_equal(actual, expected)
    assert actual[0].dtype is torch.int32
    assert actual[0].is_contiguous()
    assert actual[3].shape == (2, 2, 4)
    assert actual[3].dtype is torch.int64
    assert actual[3].is_contiguous()
    assert actual[4].dtype is torch.float32
    assert torch.equal(storage, before)
    assert actual[0].data_ptr() != labels.data_ptr()
    assert actual[1].data_ptr() != labels.data_ptr()
    assert actual[3].data_ptr() != labels.data_ptr()

    op_outputs = _prepare_input_tensors_cuda(
        labels,
        None,
        global_valid_tokens,
        True,
        True,
        1,
    )
    output_ptrs = [tensor.data_ptr() for tensor in op_outputs]
    assert len(set(output_ptrs)) == len(op_outputs)
    assert labels.data_ptr() not in output_ptrs
    assert global_valid_tokens.data_ptr() not in output_ptrs


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@_cuda_available()
def test_supplied_positions_preserve_identity_dtype_resets_and_strides(position_dtype):
    device = torch.device("cuda:0")
    labels_storage = torch.tensor(
        [
            [1, 90, 2, 91, IGNORE_INDEX, 92, 3, 93],
            [IGNORE_INDEX, 80, IGNORE_INDEX, 81, 4, 82, IGNORE_INDEX, 83],
        ],
        dtype=torch.int64,
        device=device,
    )
    labels = labels_storage[:, ::2]
    positions_storage = torch.tensor(
        [[0, 70, 1, 71, 0, 72, 1, 73], [0, 80, 0, 81, 1, 82, 2, 83]],
        dtype=position_dtype,
        device=device,
    )
    positions = positions_storage[:, ::2]
    labels_before = labels_storage.clone()
    positions_before = positions_storage.clone()
    global_valid_tokens = torch.tensor(100, dtype=torch.int64, device=device)
    args = {
        "generate_positions": False,
        "pack_mtp_targets": True,
        "global_valid_tokens": global_valid_tokens,
    }

    actual = _prepare_tensors(labels, positions, **args)
    expected = _reference(
        labels,
        positions,
        generate_positions=False,
        pack_mtp_targets=True,
    )

    _assert_preparation_equal(actual, expected)
    assert actual[0] is positions
    assert actual[0].dtype is position_dtype
    assert not actual[0].is_contiguous()
    assert actual[3].dtype is torch.int64
    assert torch.equal(actual[3][:, 1], positions.to(torch.int64))
    assert torch.equal(labels_storage, labels_before)
    assert torch.equal(positions_storage, positions_before)


@pytest.mark.parametrize(
    "shape, all_ignored",
    [((2, 0), False), ((2, 8), True), ((1, 1), False)],
)
@_cuda_available()
def test_explicit_kernel_handles_empty_all_ignored_and_sequence_edge(
    shape, all_ignored
):
    device = torch.device("cuda:0")
    labels = torch.zeros(shape, dtype=torch.int64, device=device)
    if all_ignored:
        labels.fill_(IGNORE_INDEX)
    global_valid_tokens = torch.tensor(32.0, dtype=torch.float32, device=device)
    args = {
        "generate_positions": True,
        "pack_mtp_targets": True,
        "global_valid_tokens": global_valid_tokens,
    }

    actual = _prepare_tensors(labels, None, **args)
    expected = _reference(
        labels,
        None,
        generate_positions=True,
        pack_mtp_targets=True,
    )

    _assert_preparation_equal(actual, expected)
    assert actual[0].shape == shape
    assert actual[1].shape == shape
    assert actual[2].item() == (0 if all_ignored or shape[1] == 0 else 1)
    assert actual[3].shape == (shape[0], 2, shape[1])
    assert actual[4].dtype is torch.float32


@_cuda_available()
def test_explicit_kernel_falls_back_above_crop_bound():
    device = torch.device("cuda:0")
    labels = torch.zeros(
        (1, MAX_PREPARE_ELEMENTS + 1), dtype=torch.int64, device=device
    )
    positions, mask, count, targets, share = _prepare_tensors(
        labels,
        None,
        generate_positions=True,
        pack_mtp_targets=True,
        global_valid_tokens=torch.tensor(9000, dtype=torch.int64, device=device),
    )
    expected = _reference(labels, None, generate_positions=True, pack_mtp_targets=True)

    _assert_preparation_equal((positions, mask, count, targets), expected)
    assert share is None


class _PassThroughLoss:
    def pack_targets(self, labels, extra_kwargs):
        del extra_kwargs
        return labels


class _AuxModule(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(2.0, device=device))
        declare_aux_loss(self, "test_aux")

    def apply_auxiliary(self, value):
        regularizer = self.coefficient * 4.0
        return AddAuxiliaryLoss.apply(
            value, regularizer, *aux_loss_args(self, "test_aux")
        )


def _make_trainer(model, loss_fn, aux_loss, *, cp_enabled=False, full_dtensor=False):
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.model_parts = [model]
    trainer.loss_fn = loss_fn
    trainer.aux_loss = aux_loss
    trainer.parallel_dims = SimpleNamespace(
        cp_enabled=cp_enabled,
        pp_enabled=False,
    )
    trainer.config = SimpleNamespace(
        parallelism=SimpleNamespace(full_dtensor=full_dtensor)
    )
    trainer.train_context = nullcontext
    return trainer


def _make_aux_loss(model, device, splits):
    return AuxLossTracker(
        [model],
        parallel_dims=SimpleNamespace(pp_enabled=False),
        device=device,
        microbatch_splits=splits,
    )


@_cuda_available()
def test_native_crop_aux_scaling_matches_eager_for_multiple_counts_and_denominators():
    device = torch.device("cuda:0")
    model = _AuxModule(device)
    aux_loss = _make_aux_loss(model, device, splits=3)
    trainer = _make_trainer(model, _PassThroughLoss(), aux_loss)
    reference_aux_loss = _make_aux_loss(_AuxModule(device), device, splits=3)
    scale = aux_loss._states[0].scale
    scale_identity = scale
    # The first case is the B2/S4096 count in the earlier crop check. The second
    # is the one-ULP-sensitive adjacent count from the provisional Inductor path.
    cases = (
        (7020, 100000),
        (7019, 100000),
        (8192, 2096640),
        (4096, 2096640),
        (1, 100000),
        (0, 100000),
    )
    for valid_count, global_count in cases:
        labels = torch.full((2, 4096), IGNORE_INDEX, dtype=torch.int64, device=device)
        labels.view(-1)[:valid_count] = 1
        global_valid_tokens = float(global_count)
        _, _, _, extra_kwargs = prepare_forward_batch(
            trainer,
            {"input": torch.zeros_like(labels)},
            labels,
            global_valid_tokens,
        )

        observed_count = extra_kwargs["token_mask"].sum(dtype=torch.int64)
        assert observed_count.item() == valid_count
        reference_aux_loss.weigh(
            valid_tokens=observed_count,
            global_valid_tokens=global_valid_tokens,
        )
        assert scale is scale_identity
        expected = reference_aux_loss._states[0].scale
        assert torch.equal(scale.view(torch.int32), expected.view(torch.int32))
        assert aux_loss._states[0].scale is scale_identity


@_cuda_available()
def test_two_cuda_microbatches_keep_aux_state_identity_reset_and_local_backward():
    device = torch.device("cuda:0")
    model = _AuxModule(device)
    aux_loss = _make_aux_loss(model, device, splits=2)
    trainer = _make_trainer(model, _PassThroughLoss(), aux_loss)
    scale = aux_loss._states[0].scale
    module_scale = model.aux_loss_scale

    for labels, expected_share, expected_grad in (
        (
            torch.tensor([[1, IGNORE_INDEX, 2, IGNORE_INDEX]], device=device),
            0.125,
            0.5,
        ),
        (torch.tensor([[1, 2, 3, 4]], device=device), 0.25, 1.0),
    ):
        prepare_forward_batch(
            trainer,
            {"input": torch.zeros_like(labels)},
            labels,
            torch.tensor(8, dtype=torch.int64, device=device),
        )
        assert aux_loss._states[0].scale is scale
        assert model.aux_loss_scale is module_scale is scale
        torch.testing.assert_close(
            scale, torch.tensor(expected_share, device=device), rtol=0, atol=0
        )

        model.coefficient.grad = None
        model.apply_auxiliary(torch.ones((), device=device)).backward()
        torch.testing.assert_close(
            model.coefficient.grad,
            torch.tensor(expected_grad, device=device),
            rtol=0,
            atol=0,
        )

    aux_loss.begin_step()
    torch.testing.assert_close(
        aux_loss._states[0].totals,
        torch.zeros_like(aux_loss._states[0].totals),
        rtol=0,
        atol=0,
    )


@_cuda_available()
def test_custom_op_opcheck_and_outer_compile():
    device = torch.device("cuda:0")
    labels = torch.tensor([[1, IGNORE_INDEX, 2, 3]], dtype=torch.int64, device=device)
    positions = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32, device=device)
    global_valid_tokens = torch.tensor(32.0, dtype=torch.float32, device=device)
    args = (labels, positions, global_valid_tokens, False, True, 3)

    torch.library.opcheck(_prepare_input_tensors_cuda, args)
    compiled = torch.compile(
        lambda input_labels, input_positions, global_count: _prepare_input_tensors_cuda(
            input_labels, input_positions, global_count, False, True, 3
        ),
        fullgraph=True,
    )
    actual = compiled(labels, positions, global_valid_tokens)
    eager = _prepare_input_tensors_cuda(*args)

    assert actual[0].numel() == 0
    assert actual[3].numel() > 0
    for actual_tensor, eager_tensor in zip(actual[1:], eager[1:]):
        torch.testing.assert_close(actual_tensor, eager_tensor, rtol=0, atol=0)

    scalar_args = (labels, positions, 1.0 / 32.0, False, True, 3)
    torch.library.opcheck(_prepare_input_tensors_cuda_scalar, scalar_args)
    compiled_scalar = torch.compile(
        lambda input_labels,
        input_positions,
        global_reciprocal: _prepare_input_tensors_cuda_scalar(
            input_labels,
            input_positions,
            global_reciprocal,
            False,
            True,
            3,
        ),
        fullgraph=True,
    )
    actual_scalar = compiled_scalar(labels, positions, 1.0 / 32.0)
    eager_scalar = _prepare_input_tensors_cuda_scalar(*scalar_args)
    for actual_tensor, eager_tensor in zip(actual_scalar[1:], eager_scalar[1:]):
        torch.testing.assert_close(actual_tensor, eager_tensor, rtol=0, atol=0)


class _CaptureMTPLoss(MTPLoss):
    def __init__(self):
        self.received_targets = None


class _TinyDecoder(Decoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.tensor(2.0, device="cuda:0"))
        self.attention_config = SimpleNamespace(
            mask_type=MaskType.BLOCK_CAUSAL,
            attention_mask_kind=AttentionMaskKind.BLOCK_MASK,
        )

    def get_attention_masks(self, *, positions):
        self.mask_positions = positions
        return {"document_positions": positions}


@_cuda_available()
def test_supplied_positions_reach_attention_and_mtp_targets_unchanged():
    device = torch.device("cuda:0")
    model = _TinyDecoder()
    loss_fn = _CaptureMTPLoss()
    aux_loss = _make_aux_loss(model, device, splits=1)
    trainer = _make_trainer(model, loss_fn, aux_loss)
    labels = torch.tensor([[3, 4, IGNORE_INDEX, 5]], dtype=torch.int64, device=device)
    positions = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32, device=device)
    inputs, targets, _, extra_kwargs = prepare_forward_batch(
        trainer,
        {"input": torch.ones_like(labels), "positions": positions},
        labels,
        torch.tensor(8, dtype=torch.int64, device=device),
    )

    assert inputs is not None
    assert extra_kwargs["positions"] is positions
    assert model.mask_positions is positions
    assert torch.equal(extra_kwargs["token_mask"], labels != IGNORE_INDEX)
    expected_targets = torch.stack((labels, positions.to(torch.int64)), dim=1)
    torch.testing.assert_close(targets, expected_targets, rtol=0, atol=0)


@pytest.mark.parametrize("fallback", ["context_parallel", "full_dtensor"])
def test_optional_distributed_paths_use_unchanged_preparation(fallback):
    cp_enabled = fallback == "context_parallel"
    full_dtensor = fallback == "full_dtensor"
    labels = torch.tensor([[1, IGNORE_INDEX, 2]], dtype=torch.int64)
    calls = {}

    def baseline_prepare(input_dict, baseline_labels):
        calls["input_dict"] = input_dict
        calls["labels"] = baseline_labels
        return (
            input_dict["input"],
            baseline_labels,
            {"metadata": input_dict["metadata"]},
            {
                "positions": None,
                "token_mask": baseline_labels != IGNORE_INDEX,
            },
        )

    class _Tracker:
        _splits = 1
        _states = []

        def weigh(self, *, valid_tokens, global_valid_tokens):
            calls["valid_tokens"] = valid_tokens
            calls["global_valid_tokens"] = global_valid_tokens

    trainer = _make_trainer(
        nn.Identity(),
        _PassThroughLoss(),
        _Tracker(),
        cp_enabled=cp_enabled,
        full_dtensor=full_dtensor,
    )
    trainer.post_dataloading_process = baseline_prepare
    input_dict = {"input": torch.ones_like(labels), "metadata": torch.tensor(7)}

    inputs, targets, extra_inputs, extra_kwargs = prepare_forward_batch(
        trainer, input_dict, labels, torch.tensor(2)
    )

    assert calls["input_dict"] is input_dict
    assert calls["labels"] is labels
    assert calls["valid_tokens"].item() == 2
    assert calls["global_valid_tokens"].item() == 2
    assert inputs is input_dict["input"]
    assert torch.equal(targets, labels)
    assert extra_inputs["metadata"] is input_dict["metadata"]
    assert torch.equal(extra_kwargs["token_mask"], labels != IGNORE_INDEX)


@_cuda_available()
def test_native_float_denominator_uses_explicit_kernel_and_packs_mtp(monkeypatch):
    device = torch.device("cuda:0")
    model = _TinyDecoder()
    loss_fn = _CaptureMTPLoss()
    aux_loss = _make_aux_loss(model, device, splits=1)
    trainer = _make_trainer(model, loss_fn, aux_loss)
    reference_aux_loss = _make_aux_loss(_AuxModule(device), device, splits=1)
    labels = torch.full((2, 4096), IGNORE_INDEX, dtype=torch.int64, device=device)
    labels.view(-1)[:7020] = 3
    positions = torch.arange(4096, dtype=torch.int32, device=device).repeat(2, 1)
    global_valid_tokens = 2096640.0
    scalar_op_calls = []
    original_scalar_op = input_prepare._prepare_input_tensors_cuda_scalar

    def record_scalar_op(*args):
        scalar_op_calls.append(args[2])
        return original_scalar_op(*args)

    monkeypatch.setattr(
        input_prepare, "_prepare_input_tensors_cuda_scalar", record_scalar_op
    )
    inputs, targets, _, extra_kwargs = prepare_forward_batch(
        trainer,
        {"input": torch.ones_like(labels), "positions": positions},
        labels,
        global_valid_tokens,
    )

    assert inputs is not None
    assert scalar_op_calls == [1.0 / global_valid_tokens]
    assert extra_kwargs["positions"] is positions
    assert model.mask_positions is positions
    observed_count = extra_kwargs["token_mask"].sum(dtype=torch.int64)
    assert observed_count.item() == 7020
    expected_targets = torch.stack((labels, positions.to(torch.int64)), dim=1)
    torch.testing.assert_close(targets, expected_targets, rtol=0, atol=0)
    reference_aux_loss.weigh(
        valid_tokens=observed_count,
        global_valid_tokens=global_valid_tokens,
    )
    assert torch.equal(
        aux_loss._states[0].scale.view(torch.int32),
        reference_aux_loss._states[0].scale.view(torch.int32),
    )


@_cuda_available()
def test_non_dp_float32_tensor_denominator_matches_aux_loss_tracker(monkeypatch):
    device = torch.device("cuda:0")
    model = _AuxModule(device)
    aux_loss = _make_aux_loss(model, device, splits=2)
    trainer = _make_trainer(model, _PassThroughLoss(), aux_loss)
    reference_aux_loss = _make_aux_loss(_AuxModule(device), device, splits=2)
    labels = torch.tensor([[1, IGNORE_INDEX, 2, 3]], dtype=torch.int64, device=device)
    global_valid_tokens = torch.tensor(8.0, dtype=torch.float32, device=device)
    tensor_op_calls = []
    original_tensor_op = input_prepare._prepare_input_tensors_cuda

    def record_tensor_op(*args):
        tensor_op_calls.append(args[2])
        return original_tensor_op(*args)

    monkeypatch.setattr(input_prepare, "_prepare_input_tensors_cuda", record_tensor_op)
    _, _, _, extra_kwargs = prepare_forward_batch(
        trainer,
        {"input": torch.zeros_like(labels)},
        labels,
        global_valid_tokens,
    )

    assert tensor_op_calls == [global_valid_tokens]
    observed_count = extra_kwargs["token_mask"].sum(dtype=torch.int64)
    reference_aux_loss.weigh(
        valid_tokens=observed_count,
        global_valid_tokens=global_valid_tokens,
    )
    assert torch.equal(
        aux_loss._states[0].scale.view(torch.int32),
        reference_aux_loss._states[0].scale.view(torch.int32),
    )


def test_cpu_fallback_matches_baseline_and_preserves_supplied_positions():
    labels = torch.tensor([[1, IGNORE_INDEX, 2]], dtype=torch.int64)
    positions = torch.tensor([[0, 1, 0]], dtype=torch.int32)
    global_valid_tokens = torch.tensor(8, dtype=torch.int64)
    args = {
        "generate_positions": False,
        "pack_mtp_targets": True,
        "global_valid_tokens": global_valid_tokens,
    }

    actual = _prepare_tensors(labels, positions, **args)
    expected = _reference(
        labels, positions, generate_positions=False, pack_mtp_targets=True
    )

    _assert_preparation_equal(actual, expected)
    assert actual[0] is positions
    assert actual[4] is None
