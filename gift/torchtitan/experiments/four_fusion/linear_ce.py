# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Adapted from FLA fused_linear_cross_entropy.py at
# 1d2281ff1b7988d8070f00dba5ffa9747ae3a60e, itself adapted from Liger.
"""FLA token-tiled linear/CE for GIFT's unit-upstream inner backward only.

Known per-head weights and denominator enter dlogits before its single cast.
This is deliberately not a general FLCE autograd API: the native chunk loop
immediately sums returned terms and calls backward with unit upstream.
"""

from types import MethodType

import torch
import torch.nn.functional as F
import triton
from torch.distributed.tensor import DTensor
from torch.types import Number

from torchtitan.experiments.four_fusion.stable_ce import backward_kernel, forward_kernel


@torch.library.custom_op("four_fusion::native_linear_ce", mutates_args=())
def _native_linear_ce(
    input: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    denominator: torch.Tensor | None,
    host_denominator: Number,
    head_weights: list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, hidden = input.shape
    # ATen's CUDA tensor / CPU-scalar path multiplies by reciprocal rounded to fp32.
    divisor = float(host_denominator) if denominator is None else denominator
    kernel_scale = 1.0 / divisor if denominator is None else denominator
    vocab = weight.shape[0]
    rows_per_head = rows // len(head_weights)
    # Match FLA's at-most-eight internal chunks without changing sequence chunks.
    num_tiles = min(8, triton.cdiv(vocab, hidden))
    tile_rows = triton.cdiv(rows, num_tiles)
    losses = torch.empty(rows, device=input.device, dtype=torch.float32)
    dinput = torch.empty_like(input)
    # One accumulator spans every tile and both folded main/MTP heads.
    dweight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
    for start in range(0, rows, tile_rows):
        stop = min(start + tile_rows, rows)
        tile = input[start:stop]
        targets = labels[start:stop]
        logits = F.linear(tile, weight)
        maximum = torch.empty(stop - start, device=input.device, dtype=torch.float32)
        log_denominator = torch.empty_like(maximum)
        forward_kernel[(stop - start,)](
            logits,
            targets,
            losses[start:stop],
            maximum,
            log_denominator,
            vocab,
            4096,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
        # Overwrite transient logits only after their stable statistics are ready.
        backward_kernel[(stop - start,)](
            logits,
            targets,
            maximum,
            log_denominator,
            kernel_scale,
            logits,
            vocab,
            4096,
            ROWS_PER_HEAD=rows_per_head,
            ROW_OFFSET=start,
            HEAD_WEIGHTS=tuple(head_weights),
            HOST_RECIPROCAL=denominator is None,
            num_warps=4,
            enable_fp_fusion=False,
        )
        torch.mm(logits, weight, out=dinput[start:stop])
        # Installed CUDA addmm's FP32 output support is a required runtime contract.
        torch.addmm(dweight, logits.t(), tile, out=dweight, out_dtype=torch.float32)
    terms = torch.stack(
        [
            losses[index * rows_per_head : (index + 1) * rows_per_head].sum()
            / divisor
            * weight
            for index, weight in enumerate(head_weights)
        ]
    )
    return terms, dinput, dweight.to(weight.dtype)


@_native_linear_ce.register_fake
def _native_linear_ce_fake(
    input, weight, labels, denominator, host_denominator, head_weights
):
    return (
        input.new_empty((len(head_weights),), dtype=torch.float32),
        torch.empty_like(input),
        torch.empty_like(weight),
    )


def _setup_context(ctx, inputs, output):
    _, dinput, dweight = output
    ctx.save_for_backward(dinput, dweight)
    ctx.mark_non_differentiable(dinput, dweight)


def _backward(ctx, _unit_terms_gradient, _dinput_gradient, _dweight_gradient):
    if torch.is_grad_enabled():
        raise ValueError("Native linear CE supports first-order inner backward only")
    dinput, dweight = ctx.saved_tensors
    # The caller's unit-upstream contract is structural, with no device predicate.
    return dinput, dweight, None, None, None, None


_native_linear_ce.register_autograd(_backward, setup_context=_setup_context)


def native_linear_ce(input, weight, labels, denominator, head_weights):
    """Only for immediate SUM -> unit backward in ChunkedCELoss's inner loop."""
    if not isinstance(denominator, (torch.Tensor, float, int)):
        raise ValueError(
            "Native linear CE denominator must be a Tensor or Python number"
        )
    if labels.dtype != torch.int64:
        raise ValueError("Native linear CE targets must have dtype int64")
    if labels.shape != input.shape[:-1]:
        raise ValueError("Native linear CE targets must match input.shape[:-1]")
    if (
        input.device != labels.device
        or input.device != weight.device
        or (
            isinstance(denominator, torch.Tensor) and input.device != denominator.device
        )
    ):
        raise ValueError(
            "Native linear CE input, targets, weight, and denominator must "
            "share a device"
        )
    return _native_linear_ce(
        input.reshape(-1, input.shape[-1]),
        weight,
        labels.reshape(-1),
        denominator if isinstance(denominator, torch.Tensor) else None,
        denominator if isinstance(denominator, (float, int)) else 1.0,
        head_weights,
    )[0]


class LinearCE:
    """Instance-local head adapter; preserves module __call__ and wrapper hooks."""

    def __init__(self, compile_config=None):
        self.fn = native_linear_ce
        if (
            compile_config is not None
            and compile_config.enable
            and "loss" in compile_config.components
        ):
            # Token counts vary by update; keep the host count symbolic locally.
            self.fn = torch.compile(
                self.fn, backend=compile_config.backend, fullgraph=True, dynamic=True
            )
        self.original_forward = None

    @staticmethod
    def _has_supported_primitive_forward(head, forward, implementations):
        return (
            getattr(forward, "__self__", None) is head
            and getattr(forward, "__func__", None) in implementations
        )

    @classmethod
    def _is_noop_parallel_forward(cls, head, forward, contract, implementations):
        if not isinstance(contract, tuple) or len(contract) != 3:
            return False
        wrapper, original_forward, parallel_dims = contract
        if (
            forward is not wrapper
            or head.forward is not wrapper
            or not cls._has_supported_primitive_forward(
                head, original_forward, implementations
            )
        ):
            return False

        try:
            sharding_config = head._sharding_config
            if parallel_dims.full_dtensor or sharding_config.local_map is not None:
                return False
            if any(
                not isinstance(placements, dict)
                or parallel_dims.resolve_mesh(placements.keys()) is not None
                for placements in sharding_config.state_shardings.values()
            ):
                return False

            in_src_shardings = sharding_config.in_src_shardings or {}
            in_dst_shardings = sharding_config.in_dst_shardings or {}
            if not isinstance(in_src_shardings, dict) or not isinstance(
                in_dst_shardings, dict
            ):
                return False
            for name in in_src_shardings.keys() | in_dst_shardings.keys():
                if (
                    parallel_dims.resolve_shared_mesh(
                        [in_src_shardings.get(name), in_dst_shardings.get(name)]
                    )
                    is not None
                ):
                    return False

            out_dst_shardings = sharding_config.out_dst_shardings
            if out_dst_shardings is not None and not isinstance(
                out_dst_shardings, dict
            ):
                return False
            if parallel_dims.resolve_shared_mesh([out_dst_shardings]) is not None:
                return False
        except (AssertionError, AttributeError, KeyError, TypeError, ValueError):
            return False
        return True

    def install(self, head):
        self.original_forward = head.forward
        from torchtitan.models.common.embeddings.token_debeddings import Unembedding

        implementations = (Unembedding.forward, torch.nn.Linear.forward)
        contract = getattr(head, "_parallelized_forward_contract", None)
        if contract is None:
            self.supported_head = self._has_supported_primitive_forward(
                head, self.original_forward, implementations
            )
        else:
            self.supported_head = self._is_noop_parallel_forward(
                head, self.original_forward, contract, implementations
            )
        adapter = self

        def forward(
            module, input, *, ce_labels=None, ce_head_weights=None, ce_denominator=None
        ):
            if ce_labels is None:
                return adapter.original_forward(input)
            # Executes after GFSDP __call__ has unsharded and attached input hooks.
            return adapter.fn(
                input, module.weight, ce_labels, ce_denominator, ce_head_weights
            )

        head.forward = MethodType(forward, head)

    @staticmethod
    def _host_count(denominator):
        # Native DP returns an integral float from an int64 token reduction.
        if isinstance(denominator, (float, int)) and -(1 << 63) <= denominator < (
            1 << 63
        ):
            count = int(denominator)
            if float(count) == denominator:
                return count
        return None

    @staticmethod
    def _has_logits_hooks(head):
        # GFSDP state hooks must run on scalar loss outputs. Other hooks retain logits.
        for hooks in (
            head._forward_hooks,
            head._forward_pre_hooks,
            head._backward_hooks,
            head._backward_pre_hooks,
        ):
            for hook in hooks.values():
                owner = type(getattr(hook, "__self__", None)).__module__
                module = getattr(hook, "__module__", owner)
                if "gfsdp" not in module and "torch.distributed.fsdp" not in module:
                    return True
        return False

    def chunk_loss(
        self,
        loss,
        head,
        hidden,
        labels,
        denominator,
        *,
        num_heads,
        argmax_sink,
        shape_probe,
    ):
        host_count = self._host_count(denominator)
        selected = (
            (
                denominator is None
                or isinstance(denominator, torch.Tensor)
                or host_count is not None
            )
            and self.supported_head
            and hidden.is_cuda
            and hidden.dtype == torch.bfloat16
            and hidden.requires_grad
            and not shape_probe
            and argmax_sink is None
            and not isinstance(hidden, DTensor)
            and not isinstance(labels, DTensor)
            and head.bias is None
            and getattr(head, "logit_softcap", 0.0) == 0.0
            and not self._has_logits_hooks(head)
        )
        if not selected:
            logits = head(hidden)
            if argmax_sink is not None:
                with torch.no_grad():
                    argmax_sink.append(logits.detach().argmax(dim=-1))
            return loss._chunk_loss(logits, labels, denominator, num_heads=num_heads)
        weights = [1.0] if num_heads == 1 else [1.0, *loss.loss_weights]
        if denominator is None:
            denominator = 1
        elif host_count is not None:
            # Avoid this runtime's value specialization of custom-op SymFloat args.
            denominator = host_count
        terms = head(
            hidden,
            ce_labels=labels,
            ce_head_weights=weights,
            ce_denominator=denominator,
        )
        if num_heads > 1:
            mtp_total = sum(loss.loss_weights)
            for index, (term, weight) in enumerate(zip(terms, weights)):
                detached = term.detach()
                if index == 0:
                    loss.add_component(loss.component_name, detached)
                else:
                    if mtp_total != 0.0:
                        loss.add_component("loss_mtp", detached / mtp_total)
                    if len(loss.loss_weights) > 1 and weight != 0.0:
                        loss.add_component(f"loss_mtp{index}", detached / weight)
        return terms.sum()
