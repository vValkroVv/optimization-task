"""Decoder-only language model base class.

``Decoder`` is the model-level container of transformer blocks (see
``transformer_block.py``): it owns the token embeddings, the RoPE cache, the
stack of blocks, the final norm, and the LM head, plus the shared
``forward`` / ``init_states`` / attention-mask dispatch used by most models.
"""

from typing import TypeAlias, cast

import torch
from torch.nn.attention.flex_attention import and_masks

from torchtitan.config import ParallelismConfig
from torchtitan.config.schema.components.lora import unwrap_frozen
from torchtitan.config.schema.models.common.cross_section import (
    model_fits_fp8,
    model_fits_parallelism,
    mtp_matches_loss,
    seq_len_fits_model,
)
from torchtitan.config.schema.models.common.decoder_blocks.decoder import DecoderConfig
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.config.schema.trainer import TrainerConfig
from torchtitan.models.common.attention import (
    AttentionMaskKind,
    AttentionMasksType,
    FlexAttention,
    MaskType,
    SoftmaxAttention,
    create_attention_mask,
    create_varlen_metadata_for_document,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from torchtitan.protocols.model import (
    BaseModel,
    ModelRuntimeConfig,
    materialize_runtime_model_config,
)
from torchtitan.protocols.module import ModuleDict
from torchtitan.tools.logging import logger

__all__ = ["Decoder"]


class Decoder(BaseModel):
    """Base class for autoregressive decoder-only language models."""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = DecoderConfig

    # Set by the trainer when ChunkedCELoss is used, so lm_head is applied
    # per-chunk inside the loss function instead of in forward().
    # TODO(#ISSUE): Remove after fixing PP backward to skip non-tensor
    # inputs (bool kwargs cause 'has no attribute requires_grad' errors).
    _skip_lm_head: bool = False

    def __init__(self, config: Config):
        config = cast(DecoderConfig, materialize_runtime_model_config(config))
        if config.tok_embeddings is None or config.lm_head is None:
            raise ValueError(
                "Decoder needs completed tok_embeddings and lm_head configs"
            )
        super().__init__()
        self.config = config

        config._fill_rope_contract()

        self.tok_embeddings = config.tok_embeddings.build()
        self.rope = config.rope.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        self.layers = ModuleDict()
        for i, layer_config in enumerate(config.placed_layers()):
            self.layers[str(i)] = layer_config.build()

        self.norm = config.norm.build()
        self.lm_head = config.lm_head.build()

        self.mtp = config.mtp.build() if config.mtp is not None else None

        # Resolved once at build: ``Config.attention`` walks the whole config
        # tree, which is milliseconds on a large model and must not run per step.
        self.attention_config = config.attention
        self._needs_varlen_masks = (
            self.attention_config.attention_mask_kind == AttentionMaskKind.VARLEN
        )

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        # Compute buffer_device before recursion so children (RoPE) get
        # the correct device when buffer_device is not explicitly provided.
        if buffer_device is None:
            buffer_device = self.freqs_cis.device
        super().init_states(buffer_device=buffer_device)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        assert buffer_device is None or buffer_device.type != "meta", (
            f"buffer_device must not be meta, got {buffer_device}. "
            f"Buffers should be initialized on a real device after to_empty()."
        )
        if self.rope is not None:
            # RoPE's _init_self_buffers was already called by auto-recursion
            self.freqs_cis = self.rope.cache
        else:
            # PP case: rope module was pruned, rebuild to get freqs_cis
            rope = self.config.rope.build()
            rope._init_self_buffers(buffer_device=buffer_device)
            self.freqs_cis = rope.cache

    def forward(
        self,
        tokens: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ):
        # cu_seqlens are being created here, to avoid bad slicing during PP
        if self._needs_varlen_masks and attention_masks is None:
            if positions is None:
                raise ValueError(
                    "varlen attention requires positions or attention_masks"
                )
            attention_masks = create_varlen_metadata_for_document(positions)

        # Needed for PP to carry token_embs to MTP
        mtp_carry_input_embs = self.config.mtp is not None and self.training

        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        if mtp_carry_input_embs and self.tok_embeddings is not None:
            input_embeds = h

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, positions, token_mask)

        if self.mtp is not None and self.training:
            return self._forward_mtp(
                h, input_embeds, attention_masks, positions, token_mask
            )

        if mtp_carry_input_embs and self.norm is None and self.lm_head is None:
            return h, input_embeds

        h = self.norm(h) if self.norm is not None else h

        # _skip_lm_head is an attribute rather than a forward kwarg because PP backward
        # calls .requires_grad on all stage inputs, which fails on bool kwargs.
        # TODO: fix PP backward upstream to skip non-tensor inputs
        if self._skip_lm_head:
            return h
        output = self.lm_head(h) if self.lm_head is not None else h
        return output

    def _forward_mtp(
        self,
        h: torch.Tensor,
        input_embeds: torch.Tensor | None,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training-only MTP path: stack the main next-token hidden state (post
        final norm) on top of the per-head MTP hidden states, giving
        ``[num_mtp + 1, B, S, D]`` for the shared chunked lm_head + loss (see
        ``MTPLoss``). ``input_embeds`` is the (carried) token embeddings.
        """
        assert input_embeds is not None, "MTP final stage requires input_embeds"
        assert self.mtp is not None, "MTP forward requires configured prediction heads"
        main = self.norm(h) if self.norm is not None else h
        mtp_out = self.mtp(
            h, input_embeds, self.freqs_cis, attention_masks, positions, token_mask
        )
        return torch.cat([main.unsqueeze(0), mtp_out], dim=0)

    def _get_flex_attention_masks(
        self,
        positions: torch.Tensor,
        attn_config: SoftmaxAttention.Config,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]

        match attn_config.mask_type:
            case MaskType.CAUSAL:
                B = 1
            case MaskType.BLOCK_CAUSAL:
                B = positions.shape[0]
                mask_mods.append(get_document_mask_mod(positions))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {attn_config.mask_type}"
                )

        seq_len = positions.shape[1]
        kernel = unwrap_frozen(attn_config.inner_attention)
        assert isinstance(kernel, FlexAttention.Config)
        return create_attention_mask(
            and_masks(*mask_mods),
            B,
            None,
            seq_len,
            seq_len,
            BLOCK_SIZE=kernel.block_size,
        )

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> AttentionMasksType:
        # Dispatch on the family-agnostic mask kind the attention config
        # reports, rather than enumerating concrete inner-kernel types. Any
        # attention family (softmax / linear / sparse) plugs in by declaring its
        # ``attention_mask_kind``; this method needs no per-family branches.
        attn_config = self.attention_config
        mask_kind = attn_config.attention_mask_kind
        if mask_kind == AttentionMaskKind.BLOCK_MASK:
            assert isinstance(attn_config, SoftmaxAttention.Config)
            return self._get_flex_attention_masks(positions, attn_config)
        elif mask_kind == AttentionMaskKind.VARLEN:
            if attn_config.mask_type != MaskType.BLOCK_CAUSAL:
                raise ValueError(
                    f"varlen attention is only supported with block_causal "
                    f"attention mask type, got {attn_config.mask_type}"
                )
            return create_varlen_metadata_for_document(positions)
        else:
            raise TypeError(
                f"{type(attn_config).__name__} reports attention_mask_kind="
                f"{mask_kind}; get_attention_masks should only be called when a "
                f"mask is required (BLOCK_MASK or VARLEN)."
            )

    @classmethod
    def prepare_config(
        cls,
        model_config: BaseModelConfig,
        *,
        config: ModelRuntimeConfig,
        **kwargs,
    ) -> None:
        """Apply runtime config to model config.

        When *config* is a ``Trainer.Config``, validates
        ``training.seq_len`` against the model's intrinsic
        ``rope.max_seq_len``, resizes the RoPE cache, and propagates
        debug flags. Non-trainer callers may pass any config-like
        object with a ``ParallelismConfig`` in its ``parallelism``
        field; in that case the training/debug setup is skipped.
        """
        model_config = cast(DecoderConfig, model_config)
        if not model_config.materialized:
            raise ValueError(
                f"{type(model_config).__qualname__}.prepare_config needs the "
                "materialized config (ModelSpec.model, or materialize()): "
                "edits to the written config are dropped when it is materialized"
            )

        assert hasattr(config, "parallelism"), (
            "config passed to prepare_config must provide a parallelism field."
        )
        parallelism = config.parallelism
        assert isinstance(parallelism, ParallelismConfig), (
            "config.parallelism must be a ParallelismConfig, got "
            f"{type(parallelism).__name__}."
        )

        # The checks both validation stages run, so a config that passes
        # pre-submit is not rejected here (and the reverse).
        issues = model_fits_parallelism(model_config, parallelism)
        if isinstance(config, TrainerConfig):
            issues += model_fits_fp8(model_config, config.fp8, parallelism)
            issues += seq_len_fits_model(
                model_config, config.training.seq_len, parallelism
            )
            issues += mtp_matches_loss(model_config, config.loss)
        errors = [str(issue) for issue in issues if issue.severity == "error"]
        if errors:
            raise ValueError("; ".join(errors))

        model_config._fill_rope_contract()

        # FP8 selection edits the config tree, so it runs before the build.
        fp8_config = getattr(config, "fp8", None)
        if fp8_config is not None:
            from torchtitan.components.quantization.utils import apply_float8

            apply_float8(
                model_config,
                fp8_config,
                tensor_parallel_degree=parallelism.tensor_parallel_degree,
            )

        # NOTE: Inference-only callers such as the RL generator skip
        # training.seq_len sync. Generated sequence length is not known
        # ahead of time, so keep the RoPE cache at the model's max_seq_len.
        if isinstance(config, TrainerConfig):
            debug = config.debug
            seq_len = config.training.seq_len
            # ``cache_len``, not ``max_seq_len``: the run needs only the
            # positions it can index, but the declared context must not move.
            model_config.rope = model_config.rope._copy(cache_len=seq_len)

            from torchtitan.config.schema.models.common.moe.moe import MoEConfig

            from torchtitan.config.moe import dispatcher_peak_tokens

            for moe_name, moe_cfg, _, _ in model_config.traverse(MoEConfig):
                dispatcher = moe_cfg.experts.token_dispatcher
                if dispatcher is not None and dispatcher.token_drop is not None:
                    num_tokens = dispatcher_peak_tokens(
                        local_batch_size=config.training.local_batch_size,
                        seq_len=seq_len,
                        pipeline_parallel_degree=parallelism.pipeline_parallel_degree,
                        pipeline_parallel_microbatch_size=parallelism.pipeline_parallel_microbatch_size,
                        tensor_parallel_degree=parallelism.tensor_parallel_degree,
                        context_parallel_degree=parallelism.context_parallel_degree,
                        expert_parallel_degree=parallelism.expert_parallel_degree,
                    )
                    moe_cfg.experts.token_dispatcher = dispatcher.with_capacity_context(
                        num_tokens=num_tokens,
                        top_k=moe_cfg.router.top_k,
                        num_experts=moe_cfg.router.num_experts,
                        ep_size=parallelism.expert_parallel_degree,
                    )
                    policy = dispatcher.token_drop
                    top_k = moe_cfg.router.top_k
                    num_experts = moe_cfg.router.num_experts
                    ep_size = parallelism.expert_parallel_degree
                    source_quota = policy.source_capacity(
                        num_tokens=num_tokens,
                        max_num_tokens=num_tokens,
                        top_k=top_k,
                        num_experts=num_experts,
                        ep_size=ep_size,
                    )
                    unsharded_tokens = config.training.local_batch_size * seq_len
                    factor = policy.expert_capacity_factor
                    peak_quota = (
                        int(num_tokens * top_k * ep_size * factor / num_experts)
                        // ep_size
                        if factor is not None
                        else None
                    )
                    logger.info(
                        "Token drop [%s]: strategy=%s, factor=%s, "
                        "group_gate=%s, branchless=%s; "
                        "max_tokens_per_forward=%d; "
                        "mean_routes_per_expert=%.3f; "
                        "tokens_without_PP_TP_CP=%d, "
                        "mean_routes_per_expert_without_PP_TP_CP=%.3f; "
                        "peak_factor_source_quota=%s; source_quota=%d, "
                        "EP_group_quota_per_expert=%d (EP=%d). "
                        "Quotas count token-expert routes; relative quotas "
                        "are recomputed from the actual forward size.",
                        moe_name,
                        policy.strategy,
                        factor,
                        policy.drop_by_group_capacity,
                        policy.branchless_capacity_drop,
                        num_tokens,
                        num_tokens * top_k / num_experts,
                        unsharded_tokens,
                        unsharded_tokens * top_k / num_experts,
                        peak_quota
                        if peak_quota is not None
                        else "n/a (explicit quota)",
                        source_quota,
                        source_quota * ep_size,
                        ep_size,
                    )
                if debug.moe_force_first_group:
                    num_experts = moe_cfg.router.num_experts
                    ep_size = parallelism.expert_parallel_degree
                    if (
                        num_experts % ep_size
                        or moe_cfg.router.top_k > num_experts // ep_size
                    ):
                        raise ValueError(
                            f"debug.moe_force_first_group at {moe_name} requires "
                            "num_experts divisible by expert_parallel_degree and "
                            "top_k <= num_experts // expert_parallel_degree so "
                            "only the first EP rank receives tokens; got "
                            f"top_k={moe_cfg.router.top_k}, num_experts={num_experts}, "
                            f"expert_parallel_degree={ep_size}."
                        )
                moe_cfg.router._debug_force_load_balance = debug.moe_force_load_balance
                moe_cfg.router._debug_force_first_group = debug.moe_force_first_group
