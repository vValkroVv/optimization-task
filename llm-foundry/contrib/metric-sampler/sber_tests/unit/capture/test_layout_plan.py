from __future__ import annotations

import pytest

from metric_sampler.capture.layout_plan import (
    DISPATCHER_METRICS,
    MEMORY_METRICS,
    STEP_SCALARS,
    LayoutPlan,
    LinearAttentionLayer,
    SoftmaxAttentionLayer,
)

pytestmark = pytest.mark.unit

_LOSS = ("loss/total", "loss/ce")


def _slots(plan: LayoutPlan) -> list[tuple[str, int, int, int]]:
    return [(metric.name, layer, microbatch, width) for metric, layer, microbatch, width in plan.requests()]


def _bare(**overrides) -> LayoutPlan:
    settings = dict(microbatches=1, routed_experts=4, streams=2, loss_metrics=_LOSS, token_loss=False, experts=False,
                    activations=False)
    return LayoutPlan(**{**settings, **overrides})


_STEP = [
    *((name, -1, -1, 1) for name in STEP_SCALARS),
    ("sampler/missing_slots", -1, -1, 1),
    *((name, -1, -1, 1) for name in MEMORY_METRICS),
    ("time/step_gpu_s", -1, -1, 1),
    ("time/optimizer_gpu_s", -1, -1, 1),
    ("grad/clip_total_norm", -1, -1, 1),
]


def test_a_plan_without_families_holds_the_loss_components_timings_and_step_scalars() -> None:
    assert _slots(_bare(microbatches=2)) == [
        ("loss/total", -1, 0, 1), ("loss/ce", -1, 0, 1), ("time/forward_gpu_s", -1, 0, 1),
        ("time/backward_gpu_s", -1, 0, 1),
        ("loss/total", -1, 1, 1), ("loss/ce", -1, 1, 1), ("time/forward_gpu_s", -1, 1, 1),
        ("time/backward_gpu_s", -1, 1, 1),
        *_STEP,
    ]


def test_router_counts_dispatchers_and_pre_reduce_keys_come_first_and_sorted() -> None:
    plan = _bare(counting_gate_layers=(3,), dispatcher_layers=(1,),
                 pre_reduce_keys=frozenset({("grad/pre_reduce_sq_expert", 2), ("grad/pre_reduce_sq_dense", 0)}))

    slots = _slots(plan)

    assert slots[:6] == [
        ("router/local_tokens_before_bias_update", 3, -1, 4), ("router/expert_bias_before_update", 3, -1, 4),
        *((name, 1, 0, 4) for name in DISPATCHER_METRICS),
    ]
    assert slots[-2:] == [("grad/pre_reduce_sq_dense", 0, -1, 1), ("grad/pre_reduce_sq_expert", 2, -1, 1)]


def test_dumps_add_the_skipped_dumps_counter() -> None:
    assert ("sampler/dumps_skipped", -1, -1, 1) in _slots(_bare(dumps=True))
    assert ("sampler/dumps_skipped", -1, -1, 1) not in _slots(_bare())


def test_token_loss_adds_the_microbatch_loss_statistics_with_the_detailed_ones_last() -> None:
    slots = _slots(_bare(token_loss=True, mtp_layers=(5,)))

    assert slots[4:6] == [("loss/token_max", -1, 0, 1), ("loss/nonfinite_tokens", -1, 0, 1)]
    assert slots[len(_STEP) + 6:] == [
        ("loss/token_hist_by_stream", -1, 0, 512), ("loss/mtp_token_hist", 5, 0, 256),
        ("loss/stream_sum", -1, 0, 2), ("loss/stream_sumsq", -1, 0, 2), ("loss/stream_count", -1, 0, 2),
        ("loss/stream_samples", -1, 0, 2), ("loss/position_sum", -1, 0, 64), ("loss/position_count", -1, 0, 64),
        ("loss/worst_sample_info", -1, 0, 24), ("loss/worst_sample_stats", -1, 0, 8),
        ("loss/sample_hist_by_stream", -1, 0, 256),
    ]


def test_expert_distributions_follow_the_routed_layers_only_with_experts_on() -> None:
    expert_names = {
        "router/tokens_per_expert", "router/stream_tokens_per_expert", "expert_loss/sum", "expert_loss/sumsq",
        "expert_loss/count", "expert_loss/hist", "expert_loss/stream_count", "expert_loss/stream_sum",
        "router/entropy_hist",
    }

    on = [slot for slot in _slots(_bare(experts=True, routed_layers=(2, 0))) if slot[0] in expert_names]
    off = [slot for slot in _slots(_bare(routed_layers=(2, 0))) if slot[0] in expert_names]

    assert [(name, layer) for name, layer, _, _ in on[:3]] == [("router/tokens_per_expert", 2),
                                                             ("router/stream_tokens_per_expert", 2),
                                                             ("expert_loss/sum", 2)]
    assert {(name, width) for name, layer, _, width in on if layer == 0} == {
        ("router/tokens_per_expert", 4), ("expert_loss/sum", 4), ("expert_loss/sumsq", 4), ("expert_loss/count", 4),
        ("router/stream_tokens_per_expert", 8),
        ("expert_loss/hist", 256), ("expert_loss/stream_count", 8), ("expert_loss/stream_sum", 8),
        ("router/entropy_hist", 32),
    }
    assert off == []


def test_activation_points_follow_their_layers_then_the_standalone_points() -> None:
    plan = _bare(activations=True, activation_layers={"residual": (0, 1), "down_in": (1,)})

    act = [(name, layer) for name, layer, _, _ in _slots(plan) if name.startswith("act/") and name.endswith("/moments")]

    assert act == [("act/residual/moments", 0), ("act/residual/moments", 1), ("act/down_in/moments", 1),
                   ("act/final_norm/moments", -1), ("act/embed/moments", -1)]


def test_exact_quantiles_add_a_quantile_pair_per_population() -> None:
    plan = _bare(exact_quantiles=True, token_loss=True, experts=True, activations=True, routed_layers=(1,),
                 mtp_layers=(3,), activation_layers={"residual": (0,)})

    quantiles = [slot for slot in _slots(plan) if "quantile" in slot[0]]

    assert quantiles == [
        ("loss/token_quantiles", -1, 0, 9), ("loss/token_quantile_counts", -1, 0, 5),
        ("loss/mtp_token_quantiles", 3, 0, 9), ("loss/mtp_token_quantile_counts", 3, 0, 5),
        ("router/entropy_quantiles", 1, 0, 18), ("router/entropy_quantile_counts", 1, 0, 5),
        ("act/residual/quantiles", 0, 0, 9), ("act/residual/quantile_counts", 0, 0, 5),
        ("act/final_norm/quantiles", -1, 0, 9), ("act/final_norm/quantile_counts", -1, 0, 5),
        ("act/embed/quantiles", -1, 0, 9), ("act/embed/quantile_counts", -1, 0, 5),
    ]


def test_exact_quantiles_without_activations_or_token_loss_keep_the_router_quantiles_only() -> None:
    plan = _bare(exact_quantiles=True, experts=True, routed_layers=(1,), activation_layers={"residual": (0,)})

    assert [slot for slot in _slots(plan) if "quantile" in slot[0]] == [
        ("router/entropy_quantiles", 1, 0, 18), ("router/entropy_quantile_counts", 1, 0, 5),
    ]


def test_attention_layers_take_their_slots_per_microbatch_and_linear_decay_once_per_step() -> None:
    plan = _bare(microbatches=2, softmax_attention=(SoftmaxAttentionLayer(0, heads=2, gated=False),),
                 linear_attention=(LinearAttentionLayer(1, value_heads=3),))

    attention = [slot for slot in _slots(plan) if slot[0].startswith("attn/")]

    per_microbatch = [
        ("attn/softmax/lse_stats", 0, 7), ("attn/softmax/qkv_stats", 0, 17), ("attn/softmax/row_stats", 0, 43),
        ("attn/softmax/entropy_hist", 0, 32), ("attn/softmax/entropy_quantiles", 0, 9),
        ("attn/softmax/entropy_quantile_counts", 0, 5),
        ("attn/linear/gate_stats", 1, 22), ("attn/linear/norm_stats", 1, 10), ("attn/linear/state_stats", 1, 7),
    ]
    assert attention == [
        *((name, layer, 0, width) for name, layer, width in per_microbatch),
        *((name, layer, 1, width) for name, layer, width in per_microbatch),
        ("attn/linear/decay_params", 1, -1, 6),
        ("attn/softmax/head_entropy_hist", 0, 0, 64), ("attn/softmax/head_entropy_hist", 0, 1, 64),
    ]


def test_a_gated_softmax_layer_adds_its_gate_statistics() -> None:
    plan = _bare(softmax_attention=(SoftmaxAttentionLayer(0, heads=2, gated=True),))

    names = [name for name, _, _, _ in _slots(plan) if name.startswith("attn/")]

    assert names[:4] == ["attn/softmax/lse_stats", "attn/softmax/qkv_stats", "attn/softmax/gate_stats",
                         "attn/softmax/row_stats"]
    assert ("attn/softmax/gate_stats", 0, 0, 7) in _slots(plan)


def test_extra_slots_follow_the_sampler_s_own_non_detailed_slots_and_precede_the_detailed_ones() -> None:
    plan = _bare(token_loss=True, extra=(("monitor/memory_local", -1, -1, 33),))

    slots = _slots(plan)

    assert slots[-4:] == [
        ("monitor/memory_local", -1, -1, 33), ("loss/worst_sample_info", -1, 0, 24),
        ("loss/worst_sample_stats", -1, 0, 8), ("loss/sample_hist_by_stream", -1, 0, 256),
    ]
