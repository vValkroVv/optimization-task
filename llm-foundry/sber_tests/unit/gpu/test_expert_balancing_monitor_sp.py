"""ExpertBalancingMonitor under sequence + expert parallelism.

The distributed layer is replaced with a fake that models identical data on
every rank: a SUM all_reduce multiplies by the group size, MIN/MAX are no-ops.
The tests pin down the SP-specific behaviour of the callback:

* ``fit_start`` accepts ``sp_size > 1``;
* per-expert token counts (sums over the sequence) are rescaled back to
  full-sequence values, per-token means (top-k score sums, losses) are not;
* activation norms are never reduced over the tp_sp group (whose ranks hold
  different local experts under EP).
"""
from types import SimpleNamespace

import pytest
import torch

from llmfoundry.callbacks import expert_balancing_monitor as monitor_module

pytestmark = [pytest.mark.unit, pytest.mark.requires_gpu]

SP_SIZE = 2
DP_REPLICAS = 2
WORLD = SP_SIZE * DP_REPLICAS
N_EXPERTS = 4

FSDP_GROUP = object()
TP_SP_GROUP = object()


class _SupportedModel:
    def parameters(self):
        yield torch.zeros(1)


class _State:
    def __init__(self, batch: int) -> None:
        self.model = _SupportedModel()
        self.timestamp = SimpleNamespace(batch=SimpleNamespace(value=batch))


class _Logger:
    def __init__(self) -> None:
        self.metrics: list[dict] = []

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.append(metrics)


class _Gate:
    """A gate whose router monitor returns the local (per sp shard) stats."""

    aux_loss_free = False
    alpha = 0.0

    def __init__(self, tokens, top_k_sum, scores, entropy, z_loss) -> None:
        self._tokens = torch.tensor([tokens], dtype=torch.float32)
        self._top_k_sum = torch.tensor([[top_k_sum]], dtype=torch.float32)
        self._scores = torch.tensor([scores], dtype=torch.float32)
        self._entropy = torch.tensor([[entropy]], dtype=torch.float32)
        self._z_loss = torch.tensor([[z_loss]], dtype=torch.float32)
        self.reset_calls = 0

    def activate_router_monitor(self) -> None:
        pass

    def reset_router_monitor(self) -> None:
        self.reset_calls += 1

    def get_router_monitor(self) -> dict:
        return {
            "tokens_per_layer_and_expert": self._tokens.clone(),
            "mean_top_k_scores_sum": self._top_k_sum.clone(),
            "mean_top_k_unnormed_scores_sum": self._top_k_sum.clone(),
            "mean_top_k_unnormed_biased_scores_sum": self._top_k_sum.clone(),
            "scores_per_layer_and_expert": self._scores.clone(),
            "entropy_per_layer_and_expert": self._entropy.clone(),
            "expert_biases": None,
            "z_loss_router": self._z_loss.clone(),
            "seq_aux_loss": self._z_loss.clone(),
        }


class _Experts:
    """An experts module with per-local-expert activation accumulators."""

    def __init__(self, sq_sums, counts) -> None:
        self._sq = torch.tensor(sq_sums, dtype=torch.float32)
        self._cnt = torch.tensor(counts, dtype=torch.float32)
        self.reset_calls = 0

    def activate_activation_monitor(self) -> None:
        pass

    def get_activation_norms(self):
        return self._sq.clone(), self._cnt.clone()

    def reset_activation_monitor(self) -> None:
        self.reset_calls += 1


class _FakeDist:
    def __init__(self) -> None:
        self.reduce_calls: list[tuple[object, str]] = []

    def all_reduce(self, tensor, reduce_operation="SUM", group=None, async_op=False):
        self.reduce_calls.append((group, reduce_operation))
        if reduce_operation == "SUM":
            tensor.mul_(WORLD if group in (None, FSDP_GROUP) else SP_SIZE)
        return None


def _install_fake_dist(monkeypatch: pytest.MonkeyPatch) -> _FakeDist:
    fake = _FakeDist()
    d = monitor_module.dist
    monkeypatch.setattr(d, "get_sp_group_size", lambda: SP_SIZE)
    monkeypatch.setattr(d, "get_world_size", lambda: WORLD)
    monkeypatch.setattr(d, "get_fsdp_group", lambda: FSDP_GROUP)
    monkeypatch.setattr(d, "get_fsdp_group_size", lambda: WORLD)
    monkeypatch.setattr(d, "get_tp_sp_group", lambda: TP_SP_GROUP)
    monkeypatch.setattr(d, "get_ep_group", lambda: None)
    monkeypatch.setattr(d, "all_reduce", fake.all_reduce)
    return fake


def _run(monkeypatch, gates, experts, **kwargs):
    fake = _install_fake_dist(monkeypatch)
    monkeypatch.setattr(
        monitor_module, "get_moe_gates",
        lambda state: (gates, [f"layer_{i}" for i in range(len(gates))]),
    )
    monkeypatch.setattr(
        monitor_module, "get_moe_expert_modules",
        lambda state: (experts, [f"layer_{i}" for i in range(len(experts))]),
    )
    callback = monitor_module.ExpertBalancingMonitor(batch_log_interval=1, **kwargs)
    callback.supported_models = [_SupportedModel]
    logger = _Logger()
    callback.fit_start(_State(0), logger)
    callback.batch_start(_State(0), logger)
    callback.batch_end(_State(1), logger)
    assert len(logger.metrics) == 1
    return callback, fake, logger.metrics[0]


def test_fit_start_accepts_sequence_parallelism(monkeypatch):
    callback, _, _ = _run(
        monkeypatch,
        gates=[_Gate([1.0, 1.0, 1.0, 1.0], 1.0, [0.25] * 4, 1.0, 0.1)],
        experts=[],
    )
    assert callback._sp_size == SP_SIZE


def test_router_token_counts_are_rescaled_to_full_sequence_and_means_are_not(monkeypatch):
    tokens = [10.0, 20.0, 30.0, 40.0]  # per sp shard
    gate = _Gate(tokens, top_k_sum=6.0, scores=[0.1, 0.2, 0.3, 0.4], entropy=1.2, z_loss=0.05)
    _, fake, metrics = _run(monkeypatch, gates=[gate], experts=[])

    # Every rank holds the same shard stats, so the SUM over the fsdp group is
    # x WORLD. Sums over a sequence are averaged over DP replicas only.
    assert metrics["old_moe_routing_max_load/layer_0"] == pytest.approx(40.0 * SP_SIZE)
    assert metrics["old_moe_routing_min_load/layer_0"] == pytest.approx(10.0 * SP_SIZE)
    # Per-token means stay means (the router accumulates the top-k sum per
    # token and divides by the token count).
    assert metrics["moe_routing_top_k_scores_sum/layer_0"] == pytest.approx(6.0)
    assert metrics["moe_z_loss_router/layer_0"] == pytest.approx(0.05)
    assert metrics["moe_seq_aux_loss/layer_0"] == pytest.approx(0.05)
    # Load balance uses the normalized token distribution and mean scores:
    # n_experts * sum(f_i * P_i) with f = tokens / sum(tokens).
    f = torch.tensor(tokens) / sum(tokens)
    p = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert metrics["moe_load_balance/layer_0"] == pytest.approx(N_EXPERTS * float((f * p).sum()))
    assert metrics["moe_expert_utilization/layer_0"] == pytest.approx(1.0)
    # Router stats are reduced over the fsdp group only, never over tp_sp.
    assert all(group is FSDP_GROUP for group, _ in fake.reduce_calls)
    assert gate.reset_calls == 1


def test_activation_norms_skip_tp_sp_reduce(monkeypatch):
    gate = _Gate([1.0, 1.0, 1.0, 1.0], 1.0, [0.25] * 4, 1.0, 0.1)
    # Three local experts with per-token norms sqrt(8/2) = 2, sqrt(36/4) = 3
    # and sqrt(48/3) = 4 (odd count, so the median is unambiguous).
    experts = _Experts(sq_sums=[8.0, 36.0, 48.0], counts=[2.0, 4.0, 3.0])
    _, fake, metrics = _run(
        monkeypatch, gates=[gate], experts=[experts], activate_activation_norms=True,
    )

    assert not any(group is TP_SP_GROUP for group, _ in fake.reduce_calls), (
        "activation accumulators of different local experts must not be summed over tp_sp"
    )
    assert metrics["expert_activation/mean_norm_per_token/layer_0"] == pytest.approx(3.0)
    assert metrics["expert_activation/max_median_ratio/layer_0"] == pytest.approx(4.0 / 3.0)
    assert experts.reset_calls == 1
