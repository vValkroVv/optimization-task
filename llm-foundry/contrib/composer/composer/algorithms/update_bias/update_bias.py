from __future__ import annotations

import logging
import torch
import os
from typing import Literal, Optional
import composer.utils.dist as dist
from composer.utils.model_utils import (
    cfg_get,
    get_composer_lm_backbone,
    get_lm_config,
)
from composer.core import Algorithm, Event, State, Time, TimeUnit
from composer.loggers import Logger
from composer.utils.timers import gigatimer

log = logging.getLogger(__name__)

__all__ = ["UpdateBias", "get_updated_expert_bias"]


def get_updated_expert_bias(
    tokens_per_expert: torch.Tensor,
    expert_bias: torch.Tensor,
    expert_bias_update_rate: float | torch.Tensor | list,
    logger: Logger,
    log_now: bool,
    *,
    strategy: Literal["baseline", "indicator", "threshold", "adaptive"] = "baseline",
    beta: float = 0.0,  #   (threshold/adaptive): load > avg*(1+beta)
    alpha: float = 1.0,  #    max_vio (adaptive)
    max_scale: float = 5.0,
    eps: float = 1e-6,
    name_tags: Optional[list[str]] = None,
) -> torch.Tensor:
    """
    Compute updated expert biases based on token counts.

      - "baseline": sign(avg - load) * gamma
      - "indicator": if load < avg -> +gamma; else 0
      - "threshold": if load < avg -> +gamma; else load > avg*(1+beta) -> -gamma
      - "adaptive": as "threshold", but step *= (1 + max_vio * alpha)
    """
    with torch.no_grad():
        dist.all_reduce(tokens_per_expert, reduce_operation="SUM")
        average_tokens = (
            tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        )
        dtype = expert_bias.dtype
        device = expert_bias.device
        gamma = torch.as_tensor(expert_bias_update_rate, dtype=dtype, device=device)
        # Support a per-layer update rate (one entry per stacked layer): reshape
        # to (L, 1) so it broadcasts over the expert dimension. A scalar rate is
        # left as-is (broadcasts to every layer, the original behavior).
        if gamma.ndim == 1:
            gamma = gamma.view(-1, 1)

        under = tokens_per_expert < average_tokens
        over_thr = tokens_per_expert > (average_tokens * (1.0 + beta))

        n_experts = tokens_per_expert.shape[-1]
        total_tokens = tokens_per_expert.sum(dim=-1, keepdim=True)
        expected_load = total_tokens / float(n_experts)
        max_load = tokens_per_expert.max(dim=-1, keepdim=True).values
        max_vio = (max_load - expected_load) / (expected_load + eps)

        if strategy == "baseline":
            delta = torch.sign(average_tokens - tokens_per_expert) * gamma
            scale = torch.ones_like(max_vio, dtype=dtype, device=device)
        elif strategy == "indicator":
            delta = under.to(dtype) * gamma
            scale = torch.ones_like(max_vio, dtype=dtype, device=device)
        elif strategy == "threshold":
            delta = (under.to(dtype) - over_thr.to(dtype)) * gamma
            scale = torch.ones_like(max_vio, dtype=dtype, device=device)
        elif strategy == "adaptive":
            scale = 1.0 + max_vio * torch.as_tensor(alpha, dtype=dtype, device=device)
            scales_1d = scale.squeeze(-1).detach().cpu()
            if log_now:
                metrics = {
                    f"moe_scale/layer_{i}": float(scales_1d[i].item())
                    for i in range(scales_1d.numel())
                }
                logger.log_metrics(metrics)
            scale = torch.clamp(scale, min=1.0, max=max_scale)
            delta = (under.to(dtype) - over_thr.to(dtype)) * gamma * scale
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if log_now:
            L = expert_bias.shape[0]
            if name_tags is None:
                name_tags = [f"layer_{i}" for i in range(L)]
            assert len(name_tags) == L, (
                "name_tags length must match batch (layers) dimension"
            )

            metrics = {}

            maxvio_1d = max_vio.squeeze(-1).detach().cpu()
            for i in range(L):
                metrics[f"moe_maxvio/{name_tags[i]}"] = float(maxvio_1d[i].item())

            under_frac = under.to(torch.float32).mean(dim=-1).detach().cpu()
            over_frac = over_thr.to(torch.float32).mean(dim=-1).detach().cpu()
            for i in range(L):
                metrics[f"moe_under_frac/{name_tags[i]}"] = float(under_frac[i].item())
                metrics[f"moe_over_frac/{name_tags[i]}"] = float(over_frac[i].item())

            bias_cpu = expert_bias.detach().cpu()
            bias_min = bias_cpu.min(dim=-1).values
            bias_max = bias_cpu.max(dim=-1).values
            bias_mean = bias_cpu.mean(dim=-1)
            bias_std = bias_cpu.std(dim=-1, unbiased=False)
            for i in range(L):
                nm = name_tags[i]
                metrics[f"moe_bias_min/{nm}"] = float(bias_min[i].item())
                metrics[f"moe_bias_max/{nm}"] = float(bias_max[i].item())
                metrics[f"moe_bias_mean/{nm}"] = float(bias_mean[i].item())
                metrics[f"moe_bias_std/{nm}"] = float(bias_std[i].item())
                metrics[f"moe_bias_max_min/{nm}"] = (
                    metrics[f"moe_bias_max/{nm}"] - metrics[f"moe_bias_min/{nm}"]
                )

            scales_1d = scale.squeeze(-1).detach().cpu()
            for i in range(scales_1d.numel()):
                metrics[f"moe_scale/{name_tags[i]}"] = float(scales_1d[i].item())

            logger.log_metrics(metrics)

        updated_expert_bias = expert_bias + delta
        dist.all_reduce(updated_expert_bias, reduce_operation="MAX")

        if log_now:
            L, _ = updated_expert_bias.shape
            metrics = {}
            bias_cpu = updated_expert_bias.detach().cpu()
            bias_min = bias_cpu.min(dim=-1).values
            bias_max = bias_cpu.max(dim=-1).values
            bias_mean = bias_cpu.mean(dim=-1)
            bias_std = bias_cpu.std(dim=-1, unbiased=False)
            for i in range(L):
                nm = name_tags[i]
                metrics[f"moe_bias_updated_min/{nm}"] = float(bias_min[i].item())
                metrics[f"moe_bias_updated_max/{nm}"] = float(bias_max[i].item())
                metrics[f"moe_bias_updated_mean/{nm}"] = float(bias_mean[i].item())
                metrics[f"moe_bias_updated_std/{nm}"] = float(bias_std[i].item())
                metrics[f"moe_bias_updated_max_min/{nm}"] = (
                    metrics[f"moe_bias_updated_max/{nm}"]
                    - metrics[f"moe_bias_updated_min/{nm}"]
                )
            logger.log_metrics(metrics)

        return updated_expert_bias


class UpdateBias(Algorithm):
    """
    Composer Algorithm that adjusts MoE router biases to balance expert load.
    """
    def __init__(
        self,
        moe_router_bias_update_rate: float | None = None,
        strategy: str | None = None,
        beta: float | None = None,
        alpha: float | None = None,
        scale: float | None = None,
        moe_scale_log_interval: int | str | None = None,
        bias_update_rate_scales: list | None = None,
    ):
        self._rate = moe_router_bias_update_rate
        self._strategy = strategy
        self._beta = beta
        self._alpha = alpha
        self._scale = scale

        # Static per-block multipliers on the bias-update rate, config-driven and
        # analogous to per-block LR ``lr_scales``: each entry is
        # ``{blocks: [<decoder-layer index>, ...], scale: <float>}`` and the listed
        # blocks get their update rate multiplied by ``scale``. Entries are applied
        # in order, so a later entry overrides an earlier one for any shared block.
        # ``blocks`` are REAL decoder-layer indices (``model.layers[i]``), the same
        # convention as DynamicBlockLRRescaler -- note this differs from the
        # compacted ``layer_{n}`` metric tags, which count only MoE layers. Blocks
        # with no MoE router (dense layers / out-of-range) are ignored. MTP gates
        # are never scaled. Empty => uniform global rate (original behavior).
        self._resolved_scales: dict[int, float] = {}
        for entry in (bias_update_rate_scales or []):
            blocks = entry.get("blocks")
            block_scale = entry.get("scale")
            if blocks is None or block_scale is None:
                raise ValueError(
                    f"each bias_update_rate_scales entry needs 'blocks' and 'scale'; got {entry}"
                )
            if not all(isinstance(b, int) for b in blocks):
                raise ValueError(
                    f"bias_update_rate_scales 'blocks' must be integer decoder-layer indices; got {blocks}"
                )
            if float(block_scale) <= 0.0:
                raise ValueError(
                    f"bias_update_rate_scales 'scale' must be > 0; got {block_scale}"
                )
            for b in blocks:
                self._resolved_scales[b] = float(block_scale)

        # DYNAMIC per-block multipliers, written at runtime by
        # DynamicBlockBiasRateRescaler via ``set_bias_rate_scale`` (analogous to the
        # optimizer's ``set_lr_scale``). Separate from the static config base above:
        # the effective per-block rate is ``global_rate * base * dynamic``. Kept out
        # of this algorithm's own checkpoint -- the callback owns and restores it,
        # re-applying via ``set_bias_rate_scale`` at every fit_start.
        self._dynamic_scales: dict[int, float] = {}

        if moe_scale_log_interval is None:
            self.moe_scale_log_interval = Time(20, TimeUnit.BATCH)
        elif isinstance(moe_scale_log_interval, int):
            if moe_scale_log_interval <= 0:
                raise ValueError("moe_scale_log_interval must be > 0")
            self.moe_scale_log_interval = Time(moe_scale_log_interval, TimeUnit.BATCH)
        else:
            t = Time.from_timestring(moe_scale_log_interval)
            if t.unit != TimeUnit.BATCH:
                raise ValueError("moe_scale_log_interval only supports 'ba' (batch)")
            self.moe_scale_log_interval = t

        self.enabled = False
        self.rate = 0.0
        self.strategy = "baseline"
        self.beta = 0.0
        self.alpha = 1.0
        self.scale = 5.0

    def _resolve_from_config(self, state: State):
        cfg = get_lm_config(state.model)
        if cfg is None:
            return

        aux_loss_free = cfg_get(cfg, "aux_loss_free", False)
        if not aux_loss_free:
            self.enabled = False
            return

        self.rate = (
            self._rate
            if self._rate is not None
            else cfg_get(cfg, "moe_router_bias_update_rate", None)
        )
        self.strategy = (
            self._strategy
            if self._strategy is not None
            else cfg_get(cfg, "aux_loss_free_strategy", "baseline")
        )
        self.beta = (
            self._beta if self._beta is not None else cfg_get(cfg, "adaptive_beta", 0.0)
        )
        self.alpha = (
            self._alpha
            if self._alpha is not None
            else cfg_get(cfg, "adaptive_alpha", 1.0)
        )
        self.scale = (
            self._scale
            if self._scale is not None
            else cfg_get(cfg, "adaptive_scale", 5.0)
        )

        if self.rate is None:
            raise KeyError(
                "moe_router_bias_update_rate must be set in model config or passed to UpdateBias"
            )

        self.enabled = True

        if self._resolved_scales:
            gate_blocks = self._gate_block_indices(state)
            unmatched = sorted(set(self._resolved_scales) - gate_blocks)
            if unmatched:
                log.warning(
                    "UpdateBias: bias_update_rate_scales references blocks with no "
                    f"MoE router (ignored): {unmatched}"
                )
            applied = {b: self._resolved_scales[b] for b in sorted(gate_blocks & set(self._resolved_scales))}
            log.info(f"UpdateBias scaling bias-update rate for blocks {applied} (base rate {self.rate})")

    @staticmethod
    def _gate_block_indices(state: State) -> set:
        """Real decoder-layer indices (``model.layers[i]``) that have an MoE router."""
        model = get_composer_lm_backbone(state.model)
        blocks = set()
        for i, layer in enumerate(getattr(model, "layers", [])):
            if not hasattr(layer, "self_attn") or not getattr(layer.self_attn, "is_gate", False):
                continue
            if getattr(layer.self_attn, "_gate", None) is not None:
                blocks.add(i)
        return blocks

    # ---- Dynamic per-block bias-rate scale API ---------------------------------
    # Mirrors the optimizer's set_lr_scale/get_base_lr_scale split so
    # DynamicBlockBiasRateRescaler can drive the bias-update rate the same way
    # DynamicBlockLRRescaler drives the LR. ``_resolved_scales`` is the STATIC config
    # base; ``_dynamic_scales`` is the DYNAMIC runtime multiplier. Effective per-block
    # rate = ``global_rate * base * dynamic``. Blocks are REAL decoder-layer indices.

    def gate_block_indices(self, state: State) -> set:
        """Public: real decoder-layer indices (``model.layers[i]``) with an MoE router."""
        return self._gate_block_indices(state)

    def set_bias_rate_scale(self, block: int, scale: float) -> None:
        """Set the DYNAMIC bias-update-rate multiplier for decoder layer ``block``."""
        self._dynamic_scales[int(block)] = float(scale)

    def get_bias_rate_scale(self, block: int) -> float:
        """Current DYNAMIC multiplier for ``block`` (1.0 if unset)."""
        return self._dynamic_scales.get(int(block), 1.0)

    def get_base_bias_rate_scale(self, block: int) -> float:
        """STATIC config-base multiplier for ``block`` (1.0 if unset)."""
        return self._resolved_scales.get(int(block), 1.0)

    def match(self, event: Event, state: State) -> bool:
        return event in (
            Event.INIT,
            Event.BEFORE_TRAIN_BATCH,
            Event.AFTER_TRAIN_BATCH,
            Event.AFTER_BACKWARD,
        )

    @gigatimer(timer_level=0)
    def apply(self, event: Event, state: State, logger: Logger):
        if event is Event.INIT:
            self._resolve_from_config(state)
            return

        if not self.enabled:
            return

        if event == Event.AFTER_TRAIN_BATCH:
            global_batches = state.timestamp.get(TimeUnit.BATCH).value
            log_now = global_batches % self.moe_scale_log_interval.value == 0

            tokens_list: list[torch.Tensor] = []
            bias_list: list[torch.Tensor] = []
            name_tags: list[str] = []
            # Per-layer multiplier on the update rate, aligned 1:1 with the stacked
            # tokens/bias lists. Keyed by REAL decoder index for main layers; MTP
            # gates are never scaled (1.0).
            scale_list: list[float] = []

            model = get_composer_lm_backbone(state.model)
            layers = getattr(model, "layers", [])
            mtp_layers = (
                getattr(getattr(model, "mtp_block", None), "mtp_layers", []) or []
            )

            l_idx = 0
            for i, layer in enumerate(layers):
                if not hasattr(layer, "self_attn") or not getattr(
                    layer.self_attn, "is_gate", False
                ):
                    continue
                gate = getattr(layer.self_attn, "_gate", None)
                if gate is None:
                    continue
                if self.rate != 0:
                    assert layer.self_attn.super_tensor.is_trainable, (
                        f"UpdateBias cannot update a frozen layer {l_idx} "
                        f"with non-zero rate={self.rate}"
                    )
                tokens_list.append(gate.local_tokens_per_expert)
                bias_list.append(gate.expert_bias)
                # Compacted tag preserves existing metric names; scale uses real index.
                name_tags.append(f"layer_{l_idx}")
                # Effective per-block multiplier = static config base * dynamic runtime
                # scale (the latter written by DynamicBlockBiasRateRescaler).
                scale_list.append(
                    self._resolved_scales.get(i, 1.0) * self._dynamic_scales.get(i, 1.0)
                )
                l_idx += 1

            # MTP
            m_idx = 0
            for mtp in mtp_layers:
                dec = getattr(mtp, "decoder_layer", None)
                if dec is None or not hasattr(dec, "self_attn"):
                    continue
                base_attn = getattr(dec.self_attn, "base_attn", None)
                if base_attn is None or not getattr(base_attn, "is_gate", False):
                    continue
                gate = getattr(base_attn, "_gate", None)
                if gate is None:
                    continue
                if self.rate != 0:
                    assert dec.self_attn.super_tensor.is_trainable, (
                        f"UpdateBias cannot update a frozen MTP layer {m_idx} "
                        f"with non-zero rate={self.rate}"
                    )
                tokens_list.append(gate.local_tokens_per_expert)
                bias_list.append(gate.expert_bias)
                name_tags.append(f"mtp_layer_{m_idx}")
                scale_list.append(1.0)
                m_idx += 1

            if not bias_list:
                return

            stacked_tokens = torch.stack(tokens_list)
            stacked_bias = torch.stack(bias_list)

            # Uniform scalar rate when no per-block scales are configured (original
            # behavior); otherwise a per-layer rate = base rate * per-block scale
            # (static config base * dynamic runtime scale).
            if self._resolved_scales or self._dynamic_scales:
                rate = [self.rate * s for s in scale_list]
            else:
                rate = self.rate

            updated = get_updated_expert_bias(
                stacked_tokens,
                stacked_bias,
                rate,
                logger,
                log_now,
                strategy=self.strategy,
                beta=self.beta,
                alpha=self.alpha,
                max_scale=self.scale,
                name_tags=name_tags,
            )
            for tok, old_bias, new_bias in zip(tokens_list, bias_list, updated):
                tok.zero_()
                old_bias.copy_(new_bias)

        if os.getenv("DEBUG_UPDATE_BIAS") and event == Event.BEFORE_TRAIN_BATCH:
            world_size = dist.get_world_size()
            rank = dist.get_global_rank()
            for layer in get_composer_lm_backbone(state.model).layers:
                if hasattr(layer, "self_attn"):
                    if not layer.self_attn.is_gate:
                        continue
                    gate = layer.self_attn._gate

                    local_bias = gate.expert_bias.clone()
                    gathered = [torch.zeros_like(local_bias) for _ in range(world_size)]
                    dist.dist.all_gather(gathered, local_bias)
                    for r, bias in enumerate(gathered):
                        assert torch.allclose(local_bias, bias, atol=1e-6), (
                            f"Mismatch in expert_bias at layer {layer} between rank {rank} and rank {r}"
                        )

        if os.getenv("DEBUG_UPDATE_BIAS") and event == Event.AFTER_BACKWARD:
            biases = []
            for layer in get_composer_lm_backbone(state.model).layers:
                if hasattr(layer, "self_attn"):
                    if not layer.self_attn.is_gate:
                        continue
                    gate = layer.self_attn._gate

                    biases.append(gate.expert_bias.clone().detach())
            if biases:
                stacked = torch.stack(biases)
                log.debug(f"MoE expert_bias values: {stacked}")
