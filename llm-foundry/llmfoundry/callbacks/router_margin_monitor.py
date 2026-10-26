# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor top-k selection "margin" of MoE routers, per layer.

For each token the router produces per-expert scores and selects the top-k. The
*margin* of an unselected expert is how far it fell short of the selection
boundary (the K-th selected score):

    boundary = score_{(K)}                 # K-th largest selected score
    margin_e = boundary - score_e          # for unselected expert e

A small margin means a small aux-free ``expert_bias`` could push that expert into
the top-k; a large margin means a larger bias/coefficient is needed. The layer's
"native scale" is summarized by robust statistics of these margins (medians /
low percentiles, not the mean -- a long tail of very-bad experts would inflate a
mean and mislead). Comparing this scale across layers tells you the *relative*
aux-free coefficient each layer needs: ``coef_l is proportional to margin_l``.

This codebase adds ``expert_bias`` to the post-softmax/sigmoid *scores* (see
``MoEGate.forward``: ``scores = orig_scores + expert_bias``), so the score-space
margin is the one that matches ``moe_router_bias_update_rate`` units. We also log
the raw-logit-space margin for reference. Margins are computed on the PURE
(pre-bias) router scores -- the layer's native ranking -- not on the biased
selection tensor.

Metrics (per layer tag, tag = real decoder index ``layerI`` / ``mtpM``, matching
``bias_update_rate_scales`` block indices):

    router_margin/{score,logit}/median_gap/{tag}       median_t(score_K - score_{K+1})
    router_margin/{score,logit}/median_near/{tag}      median over K+1..K+M nearest
    router_margin/{score,logit}/p25_unselected/{tag}   25th pct over all unselected
    router_margin/{score,logit}/p10_unselected/{tag}   10th pct over all unselected
    router_margin/score/median_gap_ema/{tag}           EMA of the score median gap
    router_bias_rate_scale/{tag}                       ema_l / mean_l(ema)  (suggested coef)

The ``router_bias_rate_scale`` values are a *suggestion* only (logging); they are
directly usable as ``bias_update_rate_scales`` / DynamicBlockBiasRateRescaler
multipliers. This callback never modifies training.

Limitations:
  - Disabled when ``tp_size > 1`` (router logits are expert-partitioned across TP
    ranks, so a rank cannot rank all experts).
  - With ``moe_router_group_topk`` set, selection is group-limited; this monitor
    reports the ungrouped boundary (a warning is emitted).
"""

import logging
from typing import Dict, List, Tuple

import torch
from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

from llmfoundry.callbacks.topk_accuracy_monitor import find_causal_lm

__all__ = ["RouterMarginMonitor"]

log = logging.getLogger(__name__)

# Order MUST match the tensor assembled in RouterMarginMonitor._stats.
_STAT_NAMES = [
    "score/median_gap",
    "score/median_near",
    "score/p25_unselected",
    "score/p10_unselected",
    "logit/median_gap",
    "logit/median_near",
    "logit/p25_unselected",
    "logit/p10_unselected",
]
_SCORE_MEDIAN_GAP_IDX = 0  # index of score/median_gap in the stat vector

# torch.quantile errors above ~2**24 elements; stride-subsample past this.
_QUANTILE_MAX = 4_000_000


class RouterMarginMonitor(Callback):
    """Log per-layer MoE top-k selection margins (see module docstring).

    Args:
        batch_log_interval: Compute and log every this many batches.
        near_m: Number of nearest unselected experts (K+1..K+M) for the
            ``median_near`` statistic.
        ema_decay: EMA decay for the smoothed score median-gap used to derive the
            suggested per-layer ``router_bias_rate_scale`` ratio.
    """

    def __init__(
        self,
        batch_log_interval: int = 25,
        near_m: int = 4,
        ema_decay: float = 0.9,
    ):
        if batch_log_interval <= 0:
            raise ValueError("batch_log_interval must be > 0")
        if near_m < 1:
            raise ValueError("near_m must be >= 1")
        if not (0.0 <= ema_decay < 1.0):
            raise ValueError("ema_decay must be in [0, 1)")
        self.batch_log_interval = batch_log_interval
        self.near_m = near_m
        self.ema_decay = ema_decay

        self._enabled = False
        self._active = False  # compute in hooks only on logging steps
        self._handles: list = []
        # tag -> list of [8] stat tensors, one per microbatch forward this step.
        self._acc: Dict[str, List[torch.Tensor]] = {}
        self._ema: Dict[str, float] = {}

    # ---- gate discovery (mirrors the UpdateBias algorithm traversal) ----------
    @staticmethod
    def _iter_gates(state: State) -> List[Tuple[str, torch.nn.Module]]:
        model = getattr(find_causal_lm(state), "model", None)
        gates: List[Tuple[str, torch.nn.Module]] = []
        if model is None:
            return gates
        for i, layer in enumerate(getattr(model, "layers", [])):
            attn = getattr(layer, "self_attn", None)
            if attn is None or not getattr(attn, "is_gate", False):
                continue
            gate = getattr(attn, "_gate", None)
            if gate is not None:
                gates.append((f"layer{i}", gate))
        mtp_layers = (
            getattr(getattr(model, "mtp_block", None), "mtp_layers", None) or []
        )
        for m, mtp in enumerate(mtp_layers):
            dec = getattr(mtp, "decoder_layer", None)
            attn = getattr(dec, "self_attn", None)
            base_attn = getattr(attn, "base_attn", None)
            if base_attn is None or not getattr(base_attn, "is_gate", False):
                continue
            gate = getattr(base_attn, "_gate", None)
            if gate is not None:
                gates.append((f"mtp{m}", gate))
        return gates

    def fit_start(self, state: State, logger: Logger) -> None:
        gates = self._iter_gates(state)
        if not gates:
            log.warning("RouterMarginMonitor: no MoE gates found, disabling")
            return
        if any((getattr(g, "tp_size", 1) or 1) > 1 for _, g in gates):
            log.warning(
                "RouterMarginMonitor: tp_size>1, router logits are expert-"
                "partitioned across TP ranks, disabling"
            )
            return
        if any(getattr(g, "group_topk", None) for _, g in gates):
            log.warning(
                "RouterMarginMonitor: moe_router_group_topk is set; reported "
                "margins use the ungrouped top-k boundary"
            )

        for tag, gate in gates:
            self._acc[tag] = []
            # Hook the inner linear whose output IS the raw router logits.
            handle = gate.gate.register_forward_hook(self._make_hook(tag, gate))
            self._handles.append(handle)
        self._enabled = True
        log.info(f"RouterMarginMonitor enabled on {len(gates)} gates")

    def _make_hook(self, tag: str, gate: torch.nn.Module):
        def hook(module, inputs, output):
            if not self._active or not module.training:
                return
            if getattr(gate, "_is_recompute", False):
                return  # avoid double-counting activation-checkpoint recompute
            vec = self._stats(
                output,
                k=gate.top_k,
                n_experts=gate.n_routed_experts,
                scoring_func=gate.scoring_func,
            )
            if vec is not None:
                self._acc[tag].append(vec)

        return hook

    def _stats(self, logits, k: int, n_experts: int, scoring_func: str):
        with torch.no_grad():
            fl = logits.detach().reshape(-1, n_experts).float()
            if n_experts <= k:
                return None  # no unselected experts -> no margin
            if scoring_func == "softmax":
                sc = torch.softmax(fl, dim=-1)
            elif scoring_func == "sigmoid":
                sc = torch.sigmoid(fl)
            else:
                return None
            # Pure (pre-bias) ranking is identical for logits and scores
            # (softmax/sigmoid are monotonic), so sort each descending.
            lv = fl.sort(dim=-1, descending=True).values
            sv = sc.sort(dim=-1, descending=True).values
            return torch.stack([*self._one_space(sv, k), *self._one_space(lv, k)])

    def _one_space(self, v: torch.Tensor, k: int):
        boundary = v[:, k - 1 : k]  # [T, 1]  K-th selected
        gap = (boundary - v[:, k : k + 1]).squeeze(-1).median()  # K -> K+1
        near = (boundary - v[:, k : k + self.near_m]).reshape(-1).median()
        allu = (boundary - v[:, k:]).reshape(-1)
        if allu.numel() > _QUANTILE_MAX:
            allu = allu[:: (allu.numel() // _QUANTILE_MAX + 1)]
        p25 = torch.quantile(allu, 0.25)
        p10 = torch.quantile(allu, 0.10)
        return gap, near, p25, p10

    def batch_start(self, state: State, logger: Logger) -> None:
        if not self._enabled:
            return
        self._active = (state.timestamp.batch.value + 1) % self.batch_log_interval == 0
        if self._active:
            for tag in self._acc:
                self._acc[tag].clear()

    def batch_end(self, state: State, logger: Logger) -> None:
        if not self._enabled or not self._active:
            return
        self._active = False

        tags = [t for t in self._acc if self._acc[t]]
        if not tags:
            return
        # Mean over this batch's microbatch forwards -> [num_tags, 8].
        local = torch.stack([torch.stack(self._acc[t]).mean(dim=0) for t in tags])
        reduced = self._reduce_mean(local)  # mean over DP/SP ranks
        for t in tags:
            self._acc[t].clear()

        metrics: Dict[str, float] = {}
        for row, tag in enumerate(tags):
            for j, name in enumerate(_STAT_NAMES):
                metrics[f"router_margin/{name}/{tag}"] = float(reduced[row, j].item())

        # EMA of the (reduced) score median-gap -> suggested per-layer coef ratio.
        gaps = {}
        for row, tag in enumerate(tags):
            g = float(reduced[row, _SCORE_MEDIAN_GAP_IDX].item())
            prev = self._ema.get(tag)
            ema = (
                g
                if prev is None
                else self.ema_decay * prev + (1.0 - self.ema_decay) * g
            )
            self._ema[tag] = ema
            gaps[tag] = ema
            metrics[f"router_margin/score/median_gap_ema/{tag}"] = ema

        mean_ema = sum(gaps.values()) / len(gaps)
        metrics["router_margin/score/median_gap_ema/mean"] = mean_ema
        if mean_ema > 0:
            for tag, ema in gaps.items():
                metrics[f"router_bias_rate_scale/{tag}"] = ema / mean_ema

        logger.log_metrics(metrics)

    @staticmethod
    def _reduce_mean(values: torch.Tensor) -> torch.Tensor:
        """Mean of per-rank values over the SP and FSDP process groups."""
        flat = values.reshape(-1)
        count = torch.ones(1, device=flat.device, dtype=flat.dtype)
        packed = torch.cat([flat, count])
        for group in (dist.get_sp_group(), dist.get_fsdp_group()):
            if group is not None:
                dist.all_reduce(packed, reduce_operation="SUM", group=group)
        n = packed[-1]
        if float(n.item()) == 0.0:
            return values
        return (packed[:-1] / n).reshape(values.shape)

    def close(self, state: State, logger: Logger) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
