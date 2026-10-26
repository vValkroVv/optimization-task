"""Grouped-matmul (gmm) experts.

The default routed-expert implementation: all local experts are stored as a
single stacked weight and evaluated with ``torch._grouped_mm`` over the
per-expert token offsets produced by the token dispatcher.
"""

from typing import TYPE_CHECKING, TypeAlias

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.config.schema.models.common.moe.experts.gmm import GroupedExpertsConfig
from torchtitan.models.common.moe.dispatch import dispatcher_for
from torchtitan.models.common.regularization import (
    AddAuxiliaryLoss,
    aux_loss_args,
    declare_aux_loss,
)
from torchtitan.protocols.module import Module

from .swiglu import fused_swiglu

if TYPE_CHECKING:
    from torchtitan.distributed.deepep.fp8_state import FP8DispatchState

__all__ = ["GroupedExperts"]


_HIDDEN_Z_LOSS_KINDS = ("hidden_z_loss.w1", "hidden_z_loss.w3")


def _routed_row_mask(
    num_rows: int, group_sizes_E: torch.Tensor, num_tokens_E: torch.Tensor
) -> torch.Tensor:
    """Which rows of an expert-major layout hold a routed token.

    Expert ``e`` owns ``group_sizes_E[e]`` consecutive rows, the first
    ``num_tokens_E[e]`` of which are real; the rest, and any tail past the last
    group, are padding. Sync-free: no host round-trip of the counts.
    """
    ends_E = group_sizes_E.cumsum(0)
    rows_R = torch.arange(num_rows, device=ends_E.device)
    # Tail rows land on the last expert past its group, so the offset check drops them.
    expert_R = torch.searchsorted(ends_E, rows_R, right=True).clamp(
        max=ends_E.numel() - 1
    )
    offset_R = rows_R - (ends_E - group_sizes_E)[expert_R]
    return offset_R < num_tokens_E[expert_R]


def _masked_hidden_z_loss(
    h_RF: torch.Tensor, row_mask_R: torch.Tensor, coef: float
) -> torch.Tensor:
    """``coef * logsumexp(|h|)**2`` averaged over the routed rows of ``h_RF``."""
    # mask padding rows, calculate logsumexp with another path for torch.compile.
    routed_RF = torch.where(row_mask_R.unsqueeze(-1), h_RF.float(), 0.0)
    abs_RF = routed_RF.abs()
    shift_R1 = abs_RF.detach().amax(dim=-1, keepdim=True)
    sum_exp_R = (abs_RF - shift_R1).exp().sum(dim=-1)
    lse_R = sum_exp_R.log() + shift_R1.squeeze(-1)
    routed_sum = torch.where(row_mask_R, lse_R.square(), 0.0).sum()
    # A sum over routed rows, not a mean: 0 on a rank that received none.
    return coef * routed_sum / row_mask_R.sum().clamp(min=1)


class GroupedExperts(Module):
    # pyrefly: ignore [bad-override]
    Config: TypeAlias = GroupedExpertsConfig

    def __init__(self, config: Config):
        super().__init__()
        # These flags are assigned after schema construction by FP8 selection
        # and restored after __post_init__ by _copy/build. Check at the owning
        # runtime boundary before allocating parameters or importing kernels.
        if config.use_float8_grouped_gemm:
            from torchtitan.config.schema.models.common.moe.dispatch.deepep import (
                DeepEPTokenDispatcherConfig,
            )
            from torchtitan.config.schema.models.common.moe.dispatch.hybridep import (
                HybridEPTokenDispatcherConfig,
            )

            if not isinstance(
                config.token_dispatcher,
                (DeepEPTokenDispatcherConfig, HybridEPTokenDispatcherConfig),
            ):
                raise ValueError(
                    "use_float8_grouped_gemm requires a DeepEP or HybridEP "
                    f"dispatcher, got {type(config.token_dispatcher).__name__}."
                )
            if config.score_before_experts:
                raise ValueError(
                    "use_float8_grouped_gemm requires score_before_experts=False: "
                    "the fp8 experts apply the routing probs between their two "
                    "grouped GEMMs, not before dispatch."
                )
        self.num_experts = config.num_experts
        self._use_float8_grouped_gemm = config.use_float8_grouped_gemm
        assert config.dim is not None, "partial GroupedExperts.Config"
        self._sparse_fused_swiglu_quant = config.sparse_fused_swiglu_quant
        self.swiglu_limit = config.swiglu_limit
        self.swiglu_clip_counts: torch.Tensor | None = None
        # (gate, up): the order fused_swiglu chunks the w13 output in.
        self._hidden_z_loss_coefs = (
            config.w1_hidden_z_loss_coef,
            config.w3_hidden_z_loss_coef,
        )
        for kind, coef in zip(_HIDDEN_Z_LOSS_KINDS, self._hidden_z_loss_coefs):
            if coef != 0.0:
                declare_aux_loss(self, kind)
        if any(self._hidden_z_loss_coefs):
            if config.use_float8_grouped_gemm and config.sparse_fused_swiglu_quant:
                raise ValueError(
                    "GroupedExperts w1/w3 hidden z-loss needs the bf16 SwiGLU on "
                    "the fp8 path: the fused SwiGLU+quant backward hands w13 an fp8 "
                    "gradient the z-loss gradient cannot be added to. Set "
                    "fp8.sparse_fused_swiglu_quant=False."
                )
        self.w13_E2FD = nn.Parameter(
            torch.empty(config.num_experts, 2, config.hidden_dim, config.dim)
        )
        if config.use_float8_grouped_gemm:
            from torchtitan.ops.float8.grouped_gemm import GroupedGemmFp8Wrapper

            from torchtitan.ops.float8 import quantize_expert_weights  # noqa: F401

            self.handle1 = GroupedGemmFp8Wrapper(num_sms=config.deep_gemm_num_sms)
            self.handle2 = GroupedGemmFp8Wrapper(num_sms=config.deep_gemm_num_sms)
        self.w2_EDF = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        assert config.token_dispatcher is not None  # resolved in __post_init__
        self.score_before_experts = config.score_before_experts
        self.token_dispatcher = config.token_dispatcher.build()

    def _self_grouped_param_axes(self):
        """Expert weights carry the expert count as their dim-0 group axis."""
        return ((w, self.num_experts) for w in (self.w13_E2FD, self.w2_EDF))

    def _self_named_weight_slices(self) -> dict[str, list[tuple[str, tuple]]]:
        """Which half of dim 1 of ``w13_E2FD`` is w1 and which is w3."""
        if getattr(self, "w13_E2FD", None) is None:
            return {}
        return {
            "w13_E2FD": [
                ("w1_EFD", (slice(None), slice(0, 1))),
                ("w3_EFD", (slice(None), slice(1, 2))),
            ]
        }

    def _local_expert_weights(self) -> list[torch.Tensor]:
        """Local ``(E, 2F, D)`` w13 view and ``(E, D, F)`` w2 for the grouped GEMMs."""
        w13_E2FD, w2_EDF = (
            w.to_local() if isinstance(w, DTensor) else w
            for w in (self.w13_E2FD, self.w2_EDF)
        )
        num_experts, _, hidden_dim, dim = w13_E2FD.shape
        return [w13_E2FD.view(num_experts, 2 * hidden_dim, dim), w2_EDF]

    def _quantize_local_expert_weights(self) -> list[torch.Tensor]:
        """The fused local expert weights as fp8, one edge each."""
        from torchtitan.ops.float8.quantize_expert_weights import (
            quantize_expert_weight,
        )

        return [quantize_expert_weight(w) for w in self._local_expert_weights()]

    def _clip_counts(self) -> torch.Tensor | None:
        """Where the SwiGLU adds its clamp counts; None when nothing is clamped."""
        return self.swiglu_clip_counts if self.swiglu_limit > 0.0 else None

    def _apply_hidden_z_loss(
        self,
        h_R2F: torch.Tensor,
        *,
        group_sizes_E: torch.Tensor,
        num_tokens_E: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the w1/w3 hidden z-loss to the fused w13 output (value unchanged).

        Split as ``fused_swiglu`` splits it (gate first), spliced per half, and
        rejoined, so downstream ops see the same tensor.
        """
        if not self.training or not any(self._hidden_z_loss_coefs):
            return h_R2F
        row_mask_R = _routed_row_mask(h_R2F.shape[0], group_sizes_E, num_tokens_E)
        halves = []
        for kind, coef, half in zip(
            _HIDDEN_Z_LOSS_KINDS, self._hidden_z_loss_coefs, h_R2F.chunk(2, dim=-1)
        ):
            if coef != 0.0:
                loss = _masked_hidden_z_loss(half, row_mask_R, coef)
                half = AddAuxiliaryLoss.apply(half, loss, *aux_loss_args(self, kind))
            halves.append(half)
        return torch.cat(halves, dim=-1)

    def _experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine.

        Shape suffixes here describe logical grouped-mm inputs, not physical
        sharding. Under EP, E may be a local shard of experts; under TP,
        expert weights shard hidden dimensions instead; under SP, R may be a
        local token shard. Keep logical capital suffixes here to avoid encoding
        a specific parallel layout in these local tensor names.
        """
        w13_E2FD, w2_EDF = self._local_expert_weights()
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        h_R2F = torch._grouped_mm(
            x_RD.bfloat16(),
            w13_E2FD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        h_R2F = self._apply_hidden_z_loss(
            h_R2F,
            group_sizes_E=num_tokens_per_expert_E,
            num_tokens_E=num_tokens_per_expert_E,
        )
        clip_counts = self._clip_counts()
        h_RF = fused_swiglu(
            h_R2F, swiglu_limit=self.swiglu_limit, clip_counts=clip_counts
        )
        if clip_counts is not None:
            clip_counts[2].add_(
                num_tokens_per_expert_E.sum(dtype=torch.int64) * h_RF.shape[-1]
            )
        return torch._grouped_mm(
            h_RF, w2_EDF.bfloat16().transpose(-2, -1), offs=offsets_E
        ).type_as(x_RD)

    def _fp8_experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
        fp8_state: "FP8DispatchState",
    ) -> torch.Tensor:
        """Blockwise fp8 expert compute over 128-padded token groups.

        ``num_tokens_per_expert_E`` is the unpadded per-expert count; only the
        hidden z-loss reads it, to tell routed rows from padding.
        """
        from torchtitan.ops.float8.triton_kernels.fused_swiglu_quantization import (
            fused_swiglu_quantized_fn,
        )
        from torchtitan.ops.float8.triton_kernels.utils import build_m_indices

        fp8_weights = fp8_state.fp8_expert_weights
        if fp8_weights is None:
            # The backend had nothing to overlap the quantization with.
            fp8_weights = self._quantize_local_expert_weights()
        w13_E2FD, w2_EDF = fp8_weights
        # ``x_RD`` is whatever the dispatcher returned: the payload, or the edge
        # carrying its gradient. See the gradient-edge note in hybridep.py.
        use_edges = fp8_state.uses_gradient_edges
        if use_edges:
            # It is the edge carrying the gradient; the payload is off the graph.
            payload = fp8_state.padded_tokens
            assert payload is not None, (
                "a backend declaring gradient edges must put the payload in "
                "padded_tokens: the experts have nothing else to compute on"
            )
        else:
            # It is the payload, and therefore its own edge -- the first GEMM
            # needs no separate one.
            payload = x_RD
        sizes = fp8_state.padded_group_sizes_list
        m_indices_R = build_m_indices(
            fp8_state.padded_group_sizes_t, num_rows=payload.shape[0]
        )

        # fused_swiglu_quant=False return BF16 grads, so we don't need to return edge
        fused_swiglu_quant = self._sparse_fused_swiglu_quant
        fc1_output, fc1_edge = self.handle1.group_gemm(
            tensor_groups=payload,
            weights=w13_E2FD,
            grad_edge=x_RD if use_edges else None,
            tensor_groups_sizes=sizes,
            tensor_groups_sizes_cpu=fp8_state.padded_group_sizes_cpu,
            tensor_groups_sizes_padded=fp8_state.padded_group_sizes_t,
            m_indices=m_indices_R,
            fp8_grad_output=use_edges and fused_swiglu_quant,
        )
        clip_counts = self._clip_counts()
        if fused_swiglu_quant:
            intermediate_edge, intermediate = fused_swiglu_quantized_fn(
                fc1_edge,
                fc1_output,
                fp8_state.permuted_probs,
                swiglu_limit=self.swiglu_limit,
                clip_counts=clip_counts,
            )
        else:
            fc1_output = self._apply_hidden_z_loss(
                fc1_output,
                group_sizes_E=fp8_state.padded_group_sizes_t,
                num_tokens_E=num_tokens_per_expert_E,
            )
            intermediate = fused_swiglu(
                fc1_output,
                fp8_state.permuted_probs.reshape(-1, 1),
                swiglu_limit=self.swiglu_limit,
                clip_counts=clip_counts,
            )
            intermediate_edge = None
        if clip_counts is not None:
            # The kernels scan the 128-padded rows; count the routed ones.
            half_width = fc1_output.shape[-1] // 2
            clip_counts[2].add_(
                num_tokens_per_expert_E.sum(dtype=torch.int64) * half_width
            )
        routed_output_RD, output_edge = self.handle2.group_gemm(
            tensor_groups=intermediate,
            weights=w2_EDF,
            grad_edge=intermediate_edge,
            tensor_groups_sizes=sizes,
            tensor_groups_sizes_cpu=fp8_state.padded_group_sizes_cpu,
            tensor_groups_sizes_padded=fp8_state.padded_group_sizes_t,
            m_indices=m_indices_R,
            fp8_grad_output=use_edges,
        )
        if use_edges:
            fp8_state.output_fp8_edge = output_edge
        return routed_output_RD

    def forward(
        self,
        x_BLD: torch.Tensor,
        topk_scores_BLK: torch.Tensor,
        topk_expert_ids_BLK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add.

        Returns ``(B, L_local, D)``: the caller declares it sequence-sharded.

        When parallelized, ``local_map`` (from ``sharding_config``) handles
        DTensor→local conversion on entry and local→DTensor(Partial) wrapping
        on exit. The forward body operates on plain local tensors.
        """
        B, L, D = x_BLD.shape
        K = topk_scores_BLK.size(-1)
        T = B * L
        x_TD = x_BLD.view(T, D)
        topk_scores_TK = topk_scores_BLK.view(T, K)
        topk_expert_ids_TK = topk_expert_ids_BLK.view(T, K)
        dispatch_kwargs: dict[str, object] = {}
        if self._use_float8_grouped_gemm:
            # fp8 dispatch is not configured separately: the padded expert-major
            # fp8 layout exists for these grouped GEMMs, so the two are one
            # choice, made here.
            dispatch_kwargs["use_fp8_dispatch"] = True
            dispatch_kwargs["expert_weights"] = self._local_expert_weights()
        dispatch_result = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
            score_before_experts=self.score_before_experts,
            **dispatch_kwargs,
        )
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = dispatch_result
        if self._use_float8_grouped_gemm:
            fp8_state = metadata.state.fp8
            if fp8_state is None:
                raise ValueError(
                    "fp8 grouped experts require FP8 dispatch state; the "
                    f"{type(self.token_dispatcher).__name__} did not produce one."
                )
            routed_output_RD = self._fp8_experts_forward(
                routed_input_RD, num_global_tokens_per_local_expert_e, fp8_state
            )
        else:
            routed_output_RD = self._experts_forward(
                routed_input_RD, num_global_tokens_per_local_expert_e
            )
        out_TD = self.token_dispatcher.combine(routed_output_RD, metadata, x_TD)
        # Reshaped here, under local_map: the flat (T, D) view is row-major over
        # (b, l_local) and is not a contiguous slice of the global sequence, so
        # it must never surface as a DTensor.
        return out_TD.view(B, -1, D)

    def sync_combine(self) -> None:
        """Wait for an asynchronous combine, so the caller can overlap work with it."""
        self.token_dispatcher.sync_combine()

    def parallelize(self, parallel_dims) -> None:
        """Parallelize expert weights, then settle the dispatcher's EP mesh."""
        super().parallelize(parallel_dims)
        w13 = self.w13_E2FD
        if (
            any(self._hidden_z_loss_coefs)
            and isinstance(w13, DTensor)
            and any(p.is_shard(2) for p in w13.placements)
        ):
            # The logsumexp runs over the local hidden shard only.
            raise ValueError(
                "GroupedExperts w1/w3 hidden z-loss does not support experts "
                f"sharded on the hidden dim (w13_E2FD placements {w13.placements}); "
                "use expert parallelism instead of TP for the routed experts."
            )
        # TODO(@pianpwk): with spmd_types and set_current_mesh a backend could
        # read the mesh itself instead of being handed it.
        self.token_dispatcher = dispatcher_for(
            self.token_dispatcher, ep_mesh=parallel_dims.get_optional_mesh("ep")
        )
