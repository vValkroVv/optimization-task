# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

import logging

import torch
import torch.distributed as torch_dist

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

import math

__all__ = ["ExpertBalancingMonitor"]

from llmfoundry.models.giga_mix import ComposerGigaMixCausalLM
from llmfoundry.models.dpo import ComposerDPOModel
from llmfoundry.models.gigaomni import ComposerGigaOmniForCausalLM

log = logging.getLogger(__name__)


def get_moe_gates(state: State):
    required_methods = (
        "activate_router_monitor",
        "get_router_monitor",
        "reset_router_monitor",
    )

    def _assert_supports_router_monitor(gate_layer, gate_name: str) -> None:
        missing_methods = [
            method_name
            for method_name in required_methods
            if not callable(getattr(gate_layer, method_name, None))
        ]
        assert len(missing_methods) == 0, (
            f"Gate `{gate_name}` ({type(gate_layer).__name__}) does not support router monitor. "
            f"Missing methods: {missing_methods}"
        )

    if isinstance(state.model, ComposerGigaMixCausalLM):
        model = state.model.model.model
    elif isinstance(state.model, ComposerDPOModel):
        model = state.model.model.model
    elif isinstance(state.model, ComposerGigaOmniForCausalLM):
        model = state.model.model.language_model.model
    else:
        AssertionError(
            f"Not implemented logic for another models: {type(state.model)=}"
        )

    gate_layers = []
    name_tags = []

    layers = getattr(model, "layers", [])
    mtp_layers = getattr(getattr(model, "mtp_block", None), "mtp_layers", []) or []

    for i, layer in enumerate(layers):
        if not hasattr(layer, "self_attn") or not getattr(
            layer.self_attn, "is_gate", False
        ):
            continue
        gate_layer = getattr(layer.self_attn, "_gate", None)

        if gate_layer is None:
            continue
        _assert_supports_router_monitor(gate_layer, f"layer_{i}")

        gate_layers.append(gate_layer)
        name_tags.append(f"layer_{i}")

    for i, mtp in enumerate(mtp_layers):
        dec = getattr(mtp, "decoder_layer", None)
        if dec is None or not hasattr(dec, "self_attn"):
            continue
        base_attn = getattr(dec.self_attn, "base_attn", None)
        if base_attn is None or not getattr(base_attn, "is_gate", False):
            continue
        gate_layer = getattr(base_attn, "_gate", None)

        if gate_layer is None:
            continue
        _assert_supports_router_monitor(gate_layer, f"mtp_layer_{i}")

        gate_layers.append(gate_layer)
        name_tags.append(f"mtp_layer_{i}")

    assert len(gate_layers) != 0, (
        "gate layers not found (neither in layers nor in mtp_layers)"
    )
    return gate_layers, name_tags


def get_moe_expert_modules(state: State):
    if isinstance(state.model, ComposerGigaMixCausalLM):
        model = state.model.model.model
    elif isinstance(state.model, ComposerDPOModel):
        model = state.model.model.model
    elif isinstance(state.model, ComposerGigaOmniForCausalLM):
        model = state.model.model.language_model.model
    else:
        return [], []

    expert_modules = []
    expert_name_tags = []

    layers = getattr(model, "layers", [])
    for i, layer in enumerate(layers):
        moe_block = getattr(layer, "block_sparse_moe", None)
        if moe_block is not None:
            # DeepseekGMMMoeBlock / SonicGMMMoeBlock: experts on moe_block directly
            if hasattr(moe_block, "experts"):
                expert_modules.append(moe_block.experts)
                expert_name_tags.append(f"layer_{i}")
            # ScMoEBlock: experts nested as moe_block.gemm.experts
            elif hasattr(moe_block, "gemm") and hasattr(moe_block.gemm, "experts"):
                expert_modules.append(moe_block.gemm.experts)
                expert_name_tags.append(f"layer_{i}")

    return expert_modules, expert_name_tags


class ExpertBalancingMonitor(Callback):
    """Expert balancing monitor callback.

    Periodically logs routing statistics, load balance metrics, and optionally
    expert weight norms and activation norms for MoE layers.

    Args:
        batch_log_interval (int): Logging interval in batches. Defaults to 100.
        activate_load_balance (bool): Log load balance factor and utilization. Defaults to True.
        activate_expert_weight_norms (bool): Log per-expert weight norms. Defaults to False.
        activate_activation_norms (bool): Log per-expert activation norms via forward hooks. Defaults to False.
        weight_norm_log_interval (int): Separate interval for weight norm logging (heavier). Defaults to 500.
    """

    supported_models = [
        ComposerGigaMixCausalLM,
        ComposerDPOModel,
        ComposerGigaOmniForCausalLM,
    ]

    def __init__(
        self,
        batch_log_interval: int = 100,
        activate_h_balance_metrics: bool = False,
        activate_per_expert_metrics: bool = False,
        activate_per_layer_metrics: bool = True,
        activate_load_balance: bool = True,
        activate_expert_weight_norms: bool = False,
        activate_activation_norms: bool = False,
        weight_norm_log_interval: int = 500,
    ):
        self.batch_log_interval = batch_log_interval or 100
        self.activate_h_balance_metrics = activate_h_balance_metrics
        self.activate_per_expert_metrics = activate_per_expert_metrics
        self.activate_per_layer_metrics = activate_per_layer_metrics
        self.activate_load_balance = activate_load_balance
        self.activate_expert_weight_norms = activate_expert_weight_norms
        self.activate_activation_norms = activate_activation_norms
        self.weight_norm_log_interval = weight_norm_log_interval

        self.moe_layer_gates = []
        self.name_tags = []
        self._expert_modules = []
        self._expert_name_tags = []
        self._device = None
        self._sp_size = 1

    def _fit_start(self, state: State, logger: Logger):
        # The fsdp group already contains every sp rank, so the all_reduce in batch_end
        # sums over SP; only the per-expert token counts need sp_size (see batch_end).
        self._sp_size = dist.get_sp_group_size() or 1
        if self._sp_size > 1:
            log.info(
                f"ExpertBalancingMonitor: sp_size={self._sp_size}; per-expert token "
                f"counts are rescaled to full-sequence values."
            )

        self.moe_layer_gates, self.name_tags = get_moe_gates(state=state)
        self._device = next(state.model.parameters()).device

        if self.activate_expert_weight_norms or self.activate_activation_norms:
            self._expert_modules, self._expert_name_tags = get_moe_expert_modules(state)

    def fit_start(self, state: State, logger: Logger):
        self._fit_start(state, logger)

    def batch_start(self, state: State, logger: Logger):
        if ((state.timestamp.batch.value + 1) % self.batch_log_interval) != 0:
            return

        for gate_layer in self.moe_layer_gates:
            gate_layer.activate_router_monitor()

        if self.activate_activation_norms:
            for expert_module in self._expert_modules:
                expert_module.activate_activation_monitor()

        if self.activate_expert_weight_norms:
            should_log_weights = (
                state.timestamp.batch.value + 1
            ) % self.weight_norm_log_interval == 0
            if should_log_weights:
                for expert_module in self._expert_modules:
                    expert_module.activate_weight_norm_monitor()

    def batch_end(self, state: State, logger: Logger):
        if state.timestamp.batch.value % self.batch_log_interval != 0:
            return

        training_metrics = {}
        assert isinstance(state.model, tuple(self.supported_models)), (
            f"Logic not implemented for this model type. "
            f"Supported: {self.supported_models}. Got: {type(state.model).__name__}"
        )

        utilization = []
        sparsity = []
        n_experts = 0

        for gate_layer, layer_name in zip(self.moe_layer_gates, self.name_tags):
            gate_layer.activate_router_monitor()

            router_stats = gate_layer.get_router_monitor()

            tokens_per_layer_and_expert = router_stats["tokens_per_layer_and_expert"]
            mean_top_k_scores_sum = router_stats["mean_top_k_scores_sum"]
            mean_top_k_unnormed_scores_sum = router_stats[
                "mean_top_k_unnormed_scores_sum"
            ]
            mean_top_k_unnormed_biased_scores_sum = router_stats[
                "mean_top_k_unnormed_biased_scores_sum"
            ]
            scores_per_layer_and_expert = router_stats["scores_per_layer_and_expert"]
            entropy_per_layer_and_expert = router_stats["entropy_per_layer_and_expert"]
            expert_biases = router_stats["expert_biases"]
            z_loss_router = router_stats["z_loss_router"]
            seq_aux_loss = router_stats["seq_aux_loss"]

            fsdp_group = dist.get_fsdp_group()
            n_ranks = (
                dist.get_fsdp_group_size()
                if fsdp_group is not None
                else dist.get_world_size()
            )
            # Per-token means average over ranks; per-expert token counts are sums over
            # the sequence, of which a rank holds 1/sp, so they average over dp replicas.
            n_dp_replicas = n_ranks / self._sp_size
            if gate_layer.aux_loss_free:
                expert_biases_mean_over_rank = expert_biases.clone()

                # Vars to check if expert_biases are identical over ranks
                expert_biases_min_over_rank = expert_biases_mean_over_rank.clone()
                expert_biases_max_over_rank = expert_biases_mean_over_rank.clone()

                expert_biases_mean_over_rank /= n_ranks
                z_loss_router /= n_ranks
                seq_aux_loss /= n_ranks

                # fsdp_ctx_mgr = torch_dist._coalescing_manager(group=fsdp_group)
                # with fsdp_ctx_mgr:
                dist.all_reduce(
                    expert_biases_min_over_rank,
                    reduce_operation="MIN",
                    group=fsdp_group,
                    async_op=False,
                )
                dist.all_reduce(
                    expert_biases_max_over_rank,
                    reduce_operation="MAX",
                    group=fsdp_group,
                    async_op=False,
                )
                dist.all_reduce(
                    expert_biases_mean_over_rank,
                    reduce_operation="SUM",
                    group=fsdp_group,
                    async_op=False,
                )
                dist.all_reduce(
                    seq_aux_loss,
                    reduce_operation="SUM",
                    group=fsdp_group,
                    async_op=False,
                )
                dist.all_reduce(
                    z_loss_router,
                    reduce_operation="SUM",
                    group=fsdp_group,
                    async_op=False,
                )

                assert torch.equal(
                    expert_biases_min_over_rank, expert_biases_max_over_rank
                ), "biases should be the same over differnt ranks"

            tokens_per_layer_and_expert /= n_dp_replicas
            mean_top_k_scores_sum /= n_ranks
            mean_top_k_unnormed_scores_sum /= n_ranks
            mean_top_k_unnormed_biased_scores_sum /= n_ranks
            scores_per_layer_and_expert /= n_ranks
            entropy_per_layer_and_expert /= n_ranks

            # fsdp_ctx_mgr = torch_dist._coalescing_manager(group=fsdp_group)
            # with fsdp_ctx_mgr:
            dist.all_reduce(
                tokens_per_layer_and_expert,
                reduce_operation="SUM",
                group=fsdp_group,
                async_op=False,
            )
            dist.all_reduce(
                mean_top_k_scores_sum,
                reduce_operation="SUM",
                group=fsdp_group,
                async_op=False,
            )
            dist.all_reduce(
                mean_top_k_unnormed_scores_sum,
                reduce_operation="SUM",
                group=fsdp_group,
                async_op=False,
            )
            dist.all_reduce(
                mean_top_k_unnormed_biased_scores_sum,
                reduce_operation="SUM",
                group=fsdp_group,
                async_op=False,
            )
            dist.all_reduce(
                scores_per_layer_and_expert,
                reduce_operation="SUM",
                group=fsdp_group,
                async_op=False,
            )
            dist.all_reduce(
                entropy_per_layer_and_expert,
                reduce_operation="SUM",
                group=fsdp_group,
                async_op=False,
            )

            raw = tokens_per_layer_and_expert.cpu()
            n_experts = raw.shape[-1]

            max_load = raw.max().item()
            min_load = raw.min().item()

            if self.activate_per_layer_metrics:
                training_metrics.update(
                    {
                        f"old_moe_routing_max_load/{layer_name}": max_load,
                        f"old_moe_routing_min_load/{layer_name}": min_load,
                        f"moe_seq_aux/{layer_name}": getattr(gate_layer, "alpha", None),
                        f"moe_z_loss_router/{layer_name}": z_loss_router.item(),
                        f"moe_seq_aux_loss/{layer_name}": seq_aux_loss.item(),
                    }
                )

            if gate_layer.aux_loss_free:
                expert_biases_mean_over_rank_min = expert_biases_mean_over_rank.min()
                expert_biases_mean_over_rank_max = expert_biases_mean_over_rank.max()

            tokens_per_layer_and_expert = torch.nn.functional.normalize(
                tokens_per_layer_and_expert, p=1
            )
            tokens_per_layer_and_expert = tokens_per_layer_and_expert.cpu()
            mean_top_k_scores_sum = mean_top_k_scores_sum.cpu()
            mean_top_k_unnormed_scores_sum = mean_top_k_unnormed_scores_sum.cpu()
            mean_top_k_unnormed_biased_scores_sum = (
                mean_top_k_unnormed_biased_scores_sum.cpu()
            )
            scores_per_layer_and_expert = scores_per_layer_and_expert.cpu()
            entropy_per_layer_and_expert = entropy_per_layer_and_expert.cpu()
            if self.activate_per_expert_metrics:
                training_metrics.update(
                    {
                        f"moe_routing_token_balancing/{layer_name}/expert{e}": tokens_per_layer_and_expert[
                            0, e
                        ].item()
                        for e in range(tokens_per_layer_and_expert.shape[-1])
                    }
                )

            if self.activate_per_layer_metrics:
                metrics = {
                    f"moe_routing_top_k_scores_sum/{layer_name}": mean_top_k_scores_sum.item(),
                    f"moe_routing_top_k_unnormed_scores_sum/{layer_name}": mean_top_k_unnormed_scores_sum.item(),
                    f"moe_routing_top_k_unnormed_biased_scores_sum/{layer_name}": mean_top_k_unnormed_biased_scores_sum.item(),
                }
                if getattr(gate_layer, "aux_loss_free", False):
                    metrics.update(
                        {
                            f"moe_routing_expert_biases_mean_over_rank_min/{layer_name}": expert_biases_mean_over_rank_min.item(),
                            f"moe_routing_expert_biases_mean_over_rank_max/{layer_name}": expert_biases_mean_over_rank_max.item(),
                        }
                    )
                training_metrics.update(metrics)

            # aux-loss-free bias can make the mean score negative -> log() is nan
            scores = scores_per_layer_and_expert.unsqueeze(0).clamp_min(0)
            entropy_per_layer = -scores * (scores + 1e-12).log()
            utilization.append(entropy_per_layer.sum().item())
            sparsity.append(entropy_per_layer_and_expert.item())

            if self.activate_load_balance and self.activate_per_layer_metrics:
                f_i = tokens_per_layer_and_expert.squeeze(0)
                P_i = scores_per_layer_and_expert.squeeze(0)
                lb = n_experts * (f_i * P_i).sum().item()
                training_metrics[f"moe_load_balance/{layer_name}"] = lb

                experts_with_tokens = (raw.squeeze(0) > 0).sum().item()
                training_metrics[f"moe_expert_utilization/{layer_name}"] = (
                    experts_with_tokens / n_experts
                )

            gate_layer.reset_router_monitor()

        if self.activate_load_balance and self.activate_per_layer_metrics:
            lb_values = [
                training_metrics[f"moe_load_balance/{tag}"]
                for tag in self.name_tags
                if f"moe_load_balance/{tag}" in training_metrics
            ]
            util_values = [
                training_metrics[f"moe_expert_utilization/{tag}"]
                for tag in self.name_tags
                if f"moe_expert_utilization/{tag}" in training_metrics
            ]
            if lb_values:
                training_metrics["moe_load_balance/mean"] = sum(lb_values) / len(
                    lb_values
                )
            if util_values:
                training_metrics["moe_expert_utilization/mean"] = sum(
                    util_values
                ) / len(util_values)

        if self.activate_h_balance_metrics:
            if n_experts > 0:
                h_max = math.log(n_experts)
                training_metrics.update(
                    {
                        f"moe_routing_h_{name}_norm": (
                            sum(vals) / len(vals) if vals else 0.0
                        )
                        / h_max
                        for name, vals in [
                            ("utilization", utilization),
                            ("sparsity", sparsity),
                        ]
                    }
                )

        if self.activate_expert_weight_norms and self._expert_modules:
            training_metrics.update(self._compute_expert_weight_norms())

        if self.activate_activation_norms and self._expert_modules:
            training_metrics.update(self._compute_activation_norm_metrics())

        logger.log_metrics(training_metrics)

    def _compute_expert_weight_norms(self):
        metrics = {}
        for experts, name_tag in zip(self._expert_modules, self._expert_name_tags):
            weight_norms = experts.get_weight_norms()
            if weight_norms is None:
                continue
            w1_norms_list, w2_norms_list = weight_norms
            w1_norms = torch.tensor(w1_norms_list, device=self._device)
            w2_norms = torch.tensor(w2_norms_list, device=self._device)

            ep_group = dist.get_ep_group()
            if ep_group is not None:
                w1_all = [
                    torch.zeros_like(w1_norms) for _ in range(dist.get_ep_group_size())
                ]
                w2_all = [
                    torch.zeros_like(w2_norms) for _ in range(dist.get_ep_group_size())
                ]
                torch_dist.all_gather(w1_all, w1_norms, group=ep_group)
                torch_dist.all_gather(w2_all, w2_norms, group=ep_group)
                w1_norms = torch.cat(w1_all)
                w2_norms = torch.cat(w2_all)

            w1_norms_cpu = w1_norms.cpu()
            w2_norms_cpu = w2_norms.cpu()

            w1_std = w1_norms_cpu.std().item()
            w1_median = w1_norms_cpu.median().item()
            w1_min = w1_norms_cpu.min().item()
            w1_max = w1_norms_cpu.max().item()

            w2_std = w2_norms_cpu.std().item()
            w2_median = w2_norms_cpu.median().item()
            w2_min = w2_norms_cpu.min().item()
            w2_max = w2_norms_cpu.max().item()

            metrics[f"expert_weights/w1_norm_std/{name_tag}"] = w1_std
            metrics[f"expert_weights/w1_norm_max_min_ratio/{name_tag}"] = (
                w1_max / w1_min if w1_min > 0 else float("inf")
            )
            metrics[f"expert_weights/w1_norm_min_median_ratio/{name_tag}"] = (
                w1_min / w1_median if w1_median > 0 else float("inf")
            )
            metrics[f"expert_weights/w2_norm_std/{name_tag}"] = w2_std
            metrics[f"expert_weights/w2_norm_max_min_ratio/{name_tag}"] = (
                w2_max / w2_min if w2_min > 0 else float("inf")
            )
            metrics[f"expert_weights/w2_norm_min_median_ratio/{name_tag}"] = (
                w2_min / w2_median if w2_median > 0 else float("inf")
            )

            experts.reset_weight_norm_monitor()

        return metrics

    def _compute_activation_norm_metrics(self):
        metrics = {}
        for expert_module, name_tag in zip(
            self._expert_modules, self._expert_name_tags
        ):
            accum = expert_module.get_activation_norms()
            if accum is None:
                continue
            # get_activation_norms returns GPU tensors directly — no tolist →
            # tensor round-trip needed.
            sq_sums_t, counts_t = accum

            # No tp_sp reduce: under EP those ranks hold different local experts, so it
            # would sum unrelated accumulators. The ep and fsdp reduces below cover them.
            ep_group = dist.get_ep_group()
            if ep_group is not None:
                ep_size = dist.get_ep_group_size()
                sq_all = [torch.zeros_like(sq_sums_t) for _ in range(ep_size)]
                cnt_all = [torch.zeros_like(counts_t) for _ in range(ep_size)]
                torch_dist.all_gather(sq_all, sq_sums_t, group=ep_group)
                torch_dist.all_gather(cnt_all, counts_t, group=ep_group)
                sq_sums_t = torch.cat(sq_all)
                counts_t = torch.cat(cnt_all)

            fsdp_group = dist.get_fsdp_group()
            if fsdp_group is not None:
                sq_sums_t /= dist.get_fsdp_group_size()
                counts_t /= dist.get_fsdp_group_size()
            else:
                sq_sums_t /= dist.get_world_size()
                counts_t /= dist.get_world_size()
            dist.all_reduce(
                sq_sums_t, reduce_operation="SUM", group=fsdp_group, async_op=False
            )
            dist.all_reduce(
                counts_t, reduce_operation="SUM", group=fsdp_group, async_op=False
            )

            norms_t = (sq_sums_t / counts_t.clamp_min(1)).sqrt()

            norms_cpu = norms_t.cpu()
            nonzero_mask = norms_cpu > 0
            if nonzero_mask.sum() == 0:
                expert_module.reset_activation_monitor()
                continue

            norms_nonzero = norms_cpu[nonzero_mask]
            median_val = norms_nonzero.median().item()
            max_val = norms_nonzero.max().item()
            mean_val = norms_nonzero.mean().item()

            metrics[f"expert_activation/max_median_ratio/{name_tag}"] = (
                max_val / median_val if median_val > 0 else float("inf")
            )
            metrics[f"expert_activation/mean_norm_per_token/{name_tag}"] = mean_val

            expert_module.reset_activation_monitor()

        return metrics
