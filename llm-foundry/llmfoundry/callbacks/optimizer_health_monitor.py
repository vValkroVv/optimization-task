import math

import torch
import triton
import triton.language as tl

from composer.core import Callback, State
from composer.functional.triton_kernels import div_ieee, sqrt_ieee
from composer.loggers import Logger
from composer.utils import dist

__all__ = ["OptimizerHealthMonitor"]


def classify_param(fqn: str) -> str:
    rules = {
        "embed_tokens": "embeddings",
        "lm_head": "lm_head",
        "norm": "norm",  # also matches "layernorm"
        "q_proj": "attn_q",
        "k_proj": "attn_k",
        "v_proj": "attn_v",
        "o_proj": "attn_o",
        "gate_proj": "mlp_gate_up",
        "weight1": "mlp_gate_up",
        "up_proj": "mlp_up",
        "down_proj": "mlp_down",
        "weight2": "mlp_down",
    }
    return next((tag for sub, tag in rules.items() if sub in fqn), "other")


PARAM_GROUPS = [
    "embeddings",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "mlp_gate_up",
    "mlp_up",
    "mlp_down",
    "norm",
    "lm_head",
    "other",
]


_HEALTH_BLOCK = 4096


@triton.jit
def _health_partials(v_ptr, m_ptr, p_ptr, numel, eps, eps10, partials_ptr, BLOCK: tl.constexpr):
    # One block's [sqrt(v) < eps, sqrt(v) < 10 eps, sum |m| / (sqrt(v) + eps), its sum of squares,
    # sum p^2]. IEEE sqrt and division compute the same per-element values as torch.
    pid = tl.program_id(0)
    offsets = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    v = tl.load(v_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sqrt_v = sqrt_ieee(v)
    update = tl.where(mask, div_ieee(tl.abs(m), sqrt_v + eps), 0.0)
    base = partials_ptr + pid * 5
    tl.store(base + 0, tl.sum((mask & (sqrt_v < eps)).to(tl.float32)))
    tl.store(base + 1, tl.sum((mask & (sqrt_v < eps10)).to(tl.float32)))
    tl.store(base + 2, tl.sum(update))
    tl.store(base + 3, tl.sum(update * update))
    tl.store(base + 4, tl.sum(p * p))


def _health_sums(v: torch.Tensor, m: torch.Tensor, p: torch.Tensor, eps: float, wd: float) -> torch.Tensor:
    """frac_eps_count, frac_10eps_count, update_abs_sum, grad_contrib_sq_sum and wd_contrib_sq_sum
    of one parameter in float64: one pass over its state on CUDA, block partials summed in float64."""
    if not v.is_cuda:
        sqrt_v = v.float().sqrt()
        update = m.float().abs() / (sqrt_v + eps)
        return torch.stack((
            (sqrt_v < eps).sum().double(),
            (sqrt_v < 10 * eps).sum().double(),
            update.sum().double(),
            update.square().sum().double(),
            p.float().square().sum().double() * wd ** 2,
        ))
    numel = v.numel()
    programs = triton.cdiv(numel, _HEALTH_BLOCK)
    partials = torch.empty(programs, 5, dtype=torch.float32, device=v.device)
    _health_partials[(programs,)](
        v.reshape(-1), m.reshape(-1), p.reshape(-1), numel, eps, 10 * eps, partials, BLOCK=_HEALTH_BLOCK,
    )
    sums = partials.double().sum(dim=0)
    sums[4] *= wd ** 2
    return sums


def _ordinary_health_metrics(state, optimizer, param_to_group):
    """Compute the pre-sampler monitor sums with the original per-parameter reductions."""
    accum = {}
    for group_name in PARAM_GROUPS:
        accum[group_name] = {
            "numel": 0.0,
            "frac_eps_count": 0.0,
            "frac_10eps_count": 0.0,
            "update_abs_sum": 0.0,
            "grad_contrib_sq_sum": 0.0,
            "wd_contrib_sq_sum": 0.0,
        }

    for name, p in state.model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue

        group = param_to_group.get(p)
        if group is None:
            continue

        opt_state = optimizer.state.get(p)
        if opt_state is None or "exp_avg" not in opt_state:
            continue

        eps = group.get("eps", 1e-8)
        wd = group.get("weight_decay", 0.0)
        m = opt_state["exp_avg"]
        v = opt_state["exp_avg_sq"]

        a = accum[classify_param(name)]
        with torch.no_grad():
            a["numel"] += p.numel()
            sqrt_v = v.float().sqrt()
            a["frac_eps_count"] += (sqrt_v < eps).sum().item()
            a["frac_10eps_count"] += (sqrt_v < 10 * eps).sum().item()

            den = sqrt_v + eps
            abs_m = m.float().abs()
            update = abs_m / den
            a["update_abs_sum"] += update.sum().item()
            a["grad_contrib_sq_sum"] += update.square().sum().item()
            w_sq_sum = p.data.float().square().sum().item()
            a["wd_contrib_sq_sum"] += (wd**2) * w_sq_sum
            del sqrt_v, den, abs_m, update

    n_accum_values = 6
    result = torch.zeros(
        len(PARAM_GROUPS) * n_accum_values,
        device=next(state.model.parameters()).device,
        dtype=torch.float64,
    )
    for i, group_name in enumerate(PARAM_GROUPS):
        a = accum[group_name]
        offset = i * n_accum_values
        result[offset + 0] = a["numel"]
        result[offset + 1] = a["frac_eps_count"]
        result[offset + 2] = a["frac_10eps_count"]
        result[offset + 3] = a["update_abs_sum"]
        result[offset + 4] = a["grad_contrib_sq_sum"]
        result[offset + 5] = a["wd_contrib_sq_sum"]
    return result


class OptimizerHealthMonitor(Callback):
    """Monitors optimizer health metrics during training.

    Computes per-parameter-group metrics from Adam optimizer state:
    - frac_eps: fraction of elements where sqrt(v) < eps
    - frac_10eps: fraction where sqrt(v) < 10*eps
    - update_mean: mean effective update |m|/(sqrt(v)+eps)
    - Q_wd_over_grad: weight decay domination ratio

    Optionally computes epsilon sensitivity analysis.
    """

    def __init__(
        self,
        batch_log_interval: int = 100,
        compute_sensitivity: bool = False,
        alt_eps_list: list[float] | None = None,
    ):
        self.batch_log_interval = batch_log_interval
        self.compute_sensitivity = compute_sensitivity
        self.alt_eps_list = alt_eps_list or [1e-6, 1e-5]
        self._param_to_group = None
        self._optimizer = None
        self.rank_local_observer = None

    def fit_start(self, state: State, logger: Logger):
        optimizer = state.optimizers[0]
        self._param_to_group = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                self._param_to_group[p] = group
        self._optimizer = optimizer

    def batch_end(self, state: State, logger: Logger):
        should_log = state.timestamp.batch.value % self.batch_log_interval == 0
        if not should_log and self.rank_local_observer is None:
            return

        if self._param_to_group is None:
            return

        optimizer = self._optimizer
        n_accum_values = 6
        if self.rank_local_observer is None:
            # Preserve the established monitor path unless sampler observation is requested.
            all_groups_tensor = _ordinary_health_metrics(state, optimizer, self._param_to_group)
        else:
            device = next(state.model.parameters()).device
            group_numel = [0.0] * len(PARAM_GROUPS)
            # Keep rank-local capture on the fused path to avoid per-parameter host syncs.
            sums = torch.zeros(len(PARAM_GROUPS), n_accum_values - 1, device=device, dtype=torch.float64)

            for name, p in state.model.named_parameters():
                if not p.requires_grad or p.grad is None:
                    continue

                group = self._param_to_group.get(p)
                if group is None:
                    continue

                opt_state = optimizer.state.get(p)
                if opt_state is None or "exp_avg" not in opt_state:
                    continue

                eps = group.get("eps", 1e-8)
                wd = group.get("weight_decay", 0.0)
                m = opt_state["exp_avg"]
                v = opt_state["exp_avg_sq"]
                group_index = PARAM_GROUPS.index(classify_param(name))

                with torch.no_grad():
                    group_numel[group_index] += p.numel()
                    sums[group_index] += _health_sums(v, m, p.data, eps, wd)

            numel_tensor = torch.tensor(group_numel, dtype=torch.float64)
            if device.type == "cuda":
                numel_tensor = numel_tensor.pin_memory()
            all_groups_tensor = torch.cat(
                (numel_tensor.to(device, non_blocking=True)[:, None], sums), dim=1,
            ).reshape(-1)

        if self.rank_local_observer is not None:
            self.rank_local_observer(all_groups_tensor)
        if not should_log:
            return

        fsdp_group = dist.get_fsdp_group()
        dist.all_reduce(all_groups_tensor, reduce_operation="SUM", group=fsdp_group)

        if state.gigafsdp_enabled:
            hybrid_group_count = state.model.model._meta_info["hybrid_group_count"]
        else:
            hybrid_group_count = 1

        metrics = {}
        for i, group_name in enumerate(PARAM_GROUPS):
            offset = i * n_accum_values
            numel = all_groups_tensor[offset + 0].item() / hybrid_group_count
            if numel < 1:
                continue

            frac_eps_count = all_groups_tensor[offset + 1].item() / hybrid_group_count
            frac_10eps_count = all_groups_tensor[offset + 2].item() / hybrid_group_count
            update_abs_sum = all_groups_tensor[offset + 3].item() / hybrid_group_count
            grad_contrib_sq_sum = (
                all_groups_tensor[offset + 4].item() / hybrid_group_count
            )
            wd_contrib_sq_sum = (
                all_groups_tensor[offset + 5].item() / hybrid_group_count
            )

            metrics[f"optim_health/frac_eps/{group_name}"] = frac_eps_count / numel
            metrics[f"optim_health/frac_10eps/{group_name}"] = frac_10eps_count / numel
            metrics[f"optim_health/update_mean/{group_name}"] = update_abs_sum / numel

            grad_l2 = math.sqrt(grad_contrib_sq_sum)
            wd_l2 = math.sqrt(wd_contrib_sq_sum)
            Q = wd_l2 / grad_l2 if grad_l2 > 0 else float("inf")
            metrics[f"optim_health/Q_wd_over_grad/{group_name}"] = Q

        logger.log_metrics(metrics)
