"""Explicit GPU input preparation for the four-fusion experiment."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch.distributed.tensor import DTensor

from torchtitan.components.aux_loss import AuxLossTracker
from torchtitan.components.loss import IGNORE_INDEX, MTPLoss
from torchtitan.models.common.attention import AttentionMaskKind, MaskType
from torchtitan.models.common.decoder_blocks import Decoder

MAX_PREPARE_ELEMENTS = 8192


@triton.jit
def _prepare_kernel(
    labels_ptr,
    supplied_positions_ptr,
    token_mask_ptr,
    valid_tokens_ptr,
    generated_positions_ptr,
    packed_targets_ptr,
    global_valid_tokens_ptr,
    GLOBAL_RECIPROCAL,
    GLOBAL_IS_TENSOR: tl.constexpr,
    share_ptr,
    LABELS_STRIDE_0: tl.constexpr,
    LABELS_STRIDE_1: tl.constexpr,
    POSITIONS_STRIDE_0: tl.constexpr,
    POSITIONS_STRIDE_1: tl.constexpr,
    SEQUENCE: tl.constexpr,
    NUM_ELEMENTS: tl.constexpr,
    IGNORE_VALUE: tl.constexpr,
    GENERATE_POSITIONS: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
    PACK_TARGETS: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    if NUM_ELEMENTS > 0:
        in_bounds = offsets < NUM_ELEMENTS
        rows = offsets // SEQUENCE
        columns = offsets % SEQUENCE
        label_offsets = (
            rows.to(tl.int64) * LABELS_STRIDE_0 + columns.to(tl.int64) * LABELS_STRIDE_1
        )
        labels = tl.load(labels_ptr + label_offsets, in_bounds, other=IGNORE_VALUE).to(
            tl.int64
        )
        valid = in_bounds & (labels != IGNORE_VALUE)
        tl.store(token_mask_ptr + offsets, valid, in_bounds)

        if GENERATE_POSITIONS:
            positions = columns.to(tl.int32)
            tl.store(generated_positions_ptr + offsets, positions, in_bounds)
        elif HAS_POSITIONS and PACK_TARGETS:
            positions = tl.load(
                supplied_positions_ptr
                + rows.to(tl.int64) * POSITIONS_STRIDE_0
                + columns.to(tl.int64) * POSITIONS_STRIDE_1,
                in_bounds,
                other=0,
            )

        if PACK_TARGETS:
            row_offset = rows.to(tl.int64) * 2 * SEQUENCE
            column_offset = columns.to(tl.int64)
            tl.store(
                packed_targets_ptr + row_offset + column_offset,
                labels,
                in_bounds,
            )
            tl.store(
                packed_targets_ptr + row_offset + SEQUENCE + column_offset,
                positions.to(tl.int64),
                in_bounds,
            )

        local_valid_tokens = tl.sum(valid.to(tl.int64), axis=0)
    else:
        local_valid_tokens = tl.full((), 0, tl.int64)

    tl.store(valid_tokens_ptr, local_valid_tokens)

    if GLOBAL_IS_TENSOR:
        global_valid_tokens = tl.load(global_valid_tokens_ptr).to(tl.float32)
        first_share = tl.div_rn(local_valid_tokens.to(tl.float32), global_valid_tokens)
    else:
        first_share = local_valid_tokens.to(tl.float32) * GLOBAL_RECIPROCAL
    split_reciprocal = tl.full((), 1.0 / SPLITS, tl.float32)
    tl.store(share_ptr, first_share * split_reciprocal)


def _allocate_prepare_outputs(
    labels: torch.Tensor,
    supplied_positions: torch.Tensor | None,
    generate_positions: bool,
    pack_mtp_targets: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool,
]:
    batch, sequence = labels.shape
    # This Torch version does not allow optional Tensor outputs in custom-op
    # schemas, so unused outputs are fresh zero-length tensors.
    generated_positions = (
        torch.empty(
            labels.shape,
            device=labels.device,
            dtype=torch.int32,
        )
        if generate_positions
        else torch.empty((0,), device=labels.device, dtype=torch.int32)
    )
    token_mask = torch.empty(labels.shape, device=labels.device, dtype=torch.bool)
    valid_tokens = torch.empty((), device=labels.device, dtype=torch.int64)
    should_pack = pack_mtp_targets and (
        generate_positions or supplied_positions is not None
    )
    packed_targets = (
        torch.empty((batch, 2, sequence), device=labels.device, dtype=labels.dtype)
        if should_pack
        else torch.empty((0,), device=labels.device, dtype=labels.dtype)
    )
    share = torch.empty((), device=labels.device, dtype=torch.float32)
    return (
        generated_positions,
        token_mask,
        valid_tokens,
        packed_targets,
        share,
        should_pack,
    )


def _run_prepare_kernel(
    labels: torch.Tensor,
    supplied_positions: torch.Tensor | None,
    global_valid_tokens_ptr: torch.Tensor,
    global_reciprocal: float,
    global_is_tensor: bool,
    generate_positions: bool,
    pack_mtp_targets: bool,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence = labels.shape
    num_elements = batch * sequence
    (
        generated_positions,
        token_mask,
        valid_tokens,
        packed_targets,
        share,
        should_pack,
    ) = _allocate_prepare_outputs(
        labels, supplied_positions, generate_positions, pack_mtp_targets
    )
    positions_input = labels if supplied_positions is None else supplied_positions
    positions_output = generated_positions if generate_positions else token_mask
    targets_output = packed_targets if should_pack else token_mask
    block = triton.next_power_of_2(max(num_elements, 1))
    _prepare_kernel[(1,)](
        labels,
        positions_input,
        token_mask,
        valid_tokens,
        positions_output,
        targets_output,
        global_valid_tokens_ptr,
        global_reciprocal,
        global_is_tensor,
        share,
        labels.stride(0),
        labels.stride(1),
        positions_input.stride(0),
        positions_input.stride(1),
        sequence,
        num_elements,
        IGNORE_INDEX,
        generate_positions,
        supplied_positions is not None,
        should_pack,
        splits,
        block,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return generated_positions, token_mask, valid_tokens, packed_targets, share


@torch.library.custom_op(
    "four_fusion::prepare_input_tensors", mutates_args=(), device_types="cuda"
)
def _prepare_input_tensors_cuda(
    labels: torch.Tensor,
    supplied_positions: torch.Tensor | None,
    global_valid_tokens: torch.Tensor,
    generate_positions: bool,
    pack_mtp_targets: bool,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _run_prepare_kernel(
        labels,
        supplied_positions,
        global_valid_tokens,
        1.0,
        True,
        generate_positions,
        pack_mtp_targets,
        splits,
    )


@_prepare_input_tensors_cuda.register_fake
def _prepare_input_tensors_cuda_fake(
    labels,
    supplied_positions,
    global_valid_tokens,
    generate_positions,
    pack_mtp_targets,
    splits,
):
    del global_valid_tokens, splits
    return _allocate_prepare_outputs(
        labels, supplied_positions, generate_positions, pack_mtp_targets
    )[:5]


@torch.library.custom_op(
    "four_fusion::prepare_input_tensors_scalar",
    mutates_args=(),
    device_types="cuda",
)
def _prepare_input_tensors_cuda_scalar(
    labels: torch.Tensor,
    supplied_positions: torch.Tensor | None,
    global_reciprocal: float,
    generate_positions: bool,
    pack_mtp_targets: bool,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # A scalar kernel argument avoids allocating or transferring a denominator
    # tensor for the Python-float value returned by dist_sum.
    return _run_prepare_kernel(
        labels,
        supplied_positions,
        labels,
        global_reciprocal,
        False,
        generate_positions,
        pack_mtp_targets,
        splits,
    )


@_prepare_input_tensors_cuda_scalar.register_fake
def _prepare_input_tensors_cuda_scalar_fake(
    labels,
    supplied_positions,
    global_reciprocal,
    generate_positions,
    pack_mtp_targets,
    splits,
):
    del global_reciprocal, splits
    return _allocate_prepare_outputs(
        labels, supplied_positions, generate_positions, pack_mtp_targets
    )[:5]


def _prepare_tensors_eager(
    labels: torch.Tensor,
    supplied_positions: torch.Tensor | None,
    *,
    generate_positions: bool,
    pack_mtp_targets: bool,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _supports_explicit_kernel(
    labels: torch.Tensor,
    supplied_positions: torch.Tensor | None,
    global_valid_tokens: torch.Tensor | float | None,
    splits: int,
) -> bool:
    if (
        not labels.is_cuda
        or labels.ndim != 2
        or labels.dtype != torch.int64
        or labels.numel() > MAX_PREPARE_ELEMENTS
        or global_valid_tokens is None
        or splits < 1
    ):
        return False
    if isinstance(global_valid_tokens, float):
        if not math.isfinite(global_valid_tokens) or global_valid_tokens <= 0.0:
            return False
    elif isinstance(global_valid_tokens, torch.Tensor):
        if (
            global_valid_tokens.device != labels.device
            or global_valid_tokens.dtype not in (torch.float32, torch.int64)
            or global_valid_tokens.ndim != 0
        ):
            return False
    else:
        return False
    if supplied_positions is not None and (
        supplied_positions.device != labels.device
        or supplied_positions.shape != labels.shape
        or supplied_positions.dtype not in (torch.int32, torch.int64)
    ):
        return False
    return True


def _prepare_tensors(
    labels: torch.Tensor,
    supplied_positions: torch.Tensor | None,
    *,
    generate_positions: bool,
    pack_mtp_targets: bool,
    global_valid_tokens: torch.Tensor | float | None = None,
    splits: int = 1,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Use the explicit CUDA op for the crop, with eager baseline fallback."""
    if _supports_explicit_kernel(
        labels, supplied_positions, global_valid_tokens, splits
    ):
        (
            generated_output,
            token_mask,
            valid_tokens,
            packed_output,
            share,
        ) = (
            _prepare_input_tensors_cuda_scalar(
                labels,
                supplied_positions,
                1.0 / global_valid_tokens,
                generate_positions,
                pack_mtp_targets,
                splits,
            )
            if isinstance(global_valid_tokens, float)
            else _prepare_input_tensors_cuda(
                labels,
                supplied_positions,
                global_valid_tokens,
                generate_positions,
                pack_mtp_targets,
                splits,
            )
        )
        has_packed_targets = pack_mtp_targets and (
            generate_positions or supplied_positions is not None
        )
        positions = generated_output if generate_positions else supplied_positions
        targets = packed_output if has_packed_targets else labels
        return positions, token_mask, valid_tokens, targets, share

    positions, token_mask, valid_tokens, packed_targets = _prepare_tensors_eager(
        labels,
        supplied_positions,
        generate_positions=generate_positions,
        pack_mtp_targets=pack_mtp_targets,
    )
    return positions, token_mask, valid_tokens, packed_targets, None


def _use_baseline_path(trainer: object, labels: torch.Tensor) -> bool:
    """Keep distributed transforms and overridden trainer hooks on baseline."""
    parallel_dims = trainer.parallel_dims
    parallelism = trainer.config.parallelism
    if (
        parallel_dims.cp_enabled
        or parallelism.full_dtensor
        or isinstance(labels, DTensor)
        or "post_dataloading_process" in getattr(trainer, "__dict__", {})
    ):
        return True

    trainer_type = type(trainer)
    aux_loss_type = type(trainer.aux_loss)
    return not (
        trainer_type.__module__ == "torchtitan.trainer"
        and trainer_type.__name__ == "Trainer"
        and getattr(aux_loss_type, "weigh", None) is AuxLossTracker.weigh
        and "weigh" not in getattr(trainer.aux_loss, "__dict__", {})
    )


def _prepare_with_baseline(
    trainer: object,
    input_dict: dict[str, torch.Tensor],
    labels: torch.Tensor,
    global_valid_tokens: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, object]]:
    inputs, labels, extra_inputs, extra_kwargs = trainer.post_dataloading_process(
        input_dict, labels
    )
    local_labels = labels.to_local() if isinstance(labels, DTensor) else labels
    trainer.aux_loss.weigh(
        valid_tokens=(local_labels != IGNORE_INDEX).sum(),
        global_valid_tokens=global_valid_tokens,
    )
    labels = trainer.loss_fn.pack_targets(labels, extra_kwargs)
    return inputs, labels, extra_inputs, extra_kwargs


def _copy_aux_loss_share(trainer: object, share: torch.Tensor) -> None:
    for state in trainer.aux_loss._states:
        state.scale.copy_(share)


def prepare_forward_batch(
    trainer: object,
    input_dict: dict[str, torch.Tensor],
    labels: torch.Tensor,
    global_valid_tokens: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, object]]:
    """Prepare one microbatch before forward, preserving the trainer's ordering."""
    if _use_baseline_path(trainer, labels):
        return _prepare_with_baseline(trainer, input_dict, labels, global_valid_tokens)

    inputs = input_dict.get("input")
    extra_inputs = {key: value for key, value in input_dict.items() if key != "input"}
    supplied_positions = extra_inputs.pop("positions", None)
    extra_kwargs: dict[str, object] = {}

    model = trainer.model_parts[0]
    generate_positions = False
    attention_mask_kind = None
    if isinstance(model, Decoder):
        attention_config = model.attention_config
        attention_mask_kind = attention_config.attention_mask_kind
        if attention_config.mask_type == MaskType.BLOCK_CAUSAL:
            assert supplied_positions is not None, (
                "block_causal mask requires per-document positions from the dataloader"
            )
        else:
            generate_positions = True

    pack_mtp_targets = (
        isinstance(trainer.loss_fn, MTPLoss)
        and type(trainer.loss_fn).pack_targets is MTPLoss.pack_targets
    )
    positions, token_mask, local_valid_tokens, packed_targets, share = _prepare_tensors(
        labels,
        supplied_positions,
        generate_positions=generate_positions,
        pack_mtp_targets=pack_mtp_targets,
        global_valid_tokens=global_valid_tokens,
        splits=trainer.aux_loss._splits,
    )

    if attention_mask_kind == AttentionMaskKind.BLOCK_MASK:
        extra_kwargs["attention_masks"] = model.get_attention_masks(
            positions=positions,
        )

    extra_kwargs["positions"] = positions
    extra_kwargs["token_mask"] = token_mask

    if share is None:
        trainer.aux_loss.weigh(
            valid_tokens=local_valid_tokens,
            global_valid_tokens=global_valid_tokens,
        )
    else:
        _copy_aux_loss_share(trainer, share)

    if pack_mtp_targets and packed_targets is not labels:
        labels = packed_targets
    else:
        labels = trainer.loss_fn.pack_targets(labels, extra_kwargs)

    return inputs, labels, extra_inputs, extra_kwargs
