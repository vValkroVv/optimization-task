from __future__ import annotations

import pytest

from metric_sampler.contract.catalog import METRICS, metric_by_id, metric_by_name
from metric_sampler.contract.metrics import (
    ACTIVATION_COUNTS_WIDTH,
    ACTIVATION_LOG2_BINS,
    EXPERT_LOSS_BINS,
    LAYER_ACTIVATION_POINTS,
    LOSS_POSITION_BUCKETS,
    QUANTILE_COUNTS_WIDTH,
    QUANTILE_LEVELS,
    ROUTER_ENTROPY_BINS,
    SAMPLE_LOSS_BINS,
    SCHEMA_VERSION,
    STANDALONE_ACTIVATION_POINTS,
    TOKEN_LOSS_BINS,
    Family,
    HistogramBins,
    NodeMerge,
    ValueType,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("metric_id", "name", "counts_name"),
    [
        (114, "act/residual/quantiles", "act/residual/quantile_counts"),
        (116, "act/attn_out/quantiles", "act/attn_out/quantile_counts"),
        (118, "act/mlp_out/quantiles", "act/mlp_out/quantile_counts"),
        (120, "act/final_norm/quantiles", "act/final_norm/quantile_counts"),
        (122, "act/shared_expert_out/quantiles", "act/shared_expert_out/quantile_counts"),
        (124, "act/attn_residual_contribution/quantiles", "act/attn_residual_contribution/quantile_counts"),
        (126, "act/mlp_residual_contribution/quantiles", "act/mlp_residual_contribution/quantile_counts"),
        (128, "act/swiglu_in/quantiles", "act/swiglu_in/quantile_counts"),
        (130, "act/down_in/quantiles", "act/down_in/quantile_counts"),
        (132, "act/embed/quantiles", "act/embed/quantile_counts"),
        (134, "loss/token_quantiles", "loss/token_quantile_counts"),
        (136, "loss/mtp_token_quantiles", "loss/mtp_token_quantile_counts"),
        (138, "router/entropy_quantiles", "router/entropy_quantile_counts"),
    ],
)
def test_exact_quantile_registry_preserves_population_and_rank(metric_id: int, name: str, counts_name: str) -> None:
    quantiles, counts = metric_by_name(name), metric_by_name(counts_name)
    assert (quantiles.metric_id, counts.metric_id) == (metric_id, metric_id + 1)
    assert (quantiles.family, quantiles.value_type) == (
        (Family.EXPERT_COUNTS, ValueType.INT32) if metric_id == 138
        else (Family.ROUTER_VECTORS, ValueType.FLOAT32)
    )
    assert (counts.family, counts.value_type) == (Family.EXPERT_COUNTS, ValueType.INT32)
    for metric in (quantiles, counts):
        assert metric.node_merge is NodeMerge.PER_RANK
        assert metric.required and not metric.detailed and not metric.job_identical
        assert metric.bins is None and not metric.summarize_experts
    assert "without interpolation, sampling or histogram estimation" in quantiles.description
    assert "must not be averaged" in quantiles.description
    assert "[finite, NaN, positive infinity, negative infinity, masked]" in counts.description
    if metric_id == 138:
        assert "stored losslessly as low/high UInt32 pairs, width=18" in quantiles.description


def test_quantile_levels_and_population_counts_width() -> None:
    assert QUANTILE_LEVELS == (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    assert QUANTILE_COUNTS_WIDTH == 5


@pytest.mark.parametrize(
    ("metric_id", "name", "family", "value_type"),
    [
        (140, "monitor/router_margin_local", Family.ROUTER_VECTORS, ValueType.FLOAT32),
        (141, "monitor/optimizer_health_local", Family.EXPERT_COUNTS, ValueType.INT32),
        (142, "monitor/update_weight_local", Family.EXPERT_COUNTS, ValueType.INT32),
        (143, "monitor/topk_local_counts", Family.EXPERT_COUNTS, ValueType.INT32),
        (144, "monitor/activation_local", Family.ROUTER_VECTORS, ValueType.FLOAT32),
        (145, "monitor/weight_local", Family.ROUTER_VECTORS, ValueType.FLOAT32),
        (146, "monitor/activation_local_observations", Family.EXPERT_COUNTS, ValueType.INT32),
        (147, "monitor/training_local_counts", Family.EXPERT_COUNTS, ValueType.INT32),
        (149, "monitor/optimizer_local", Family.ROUTER_VECTORS, ValueType.FLOAT32),
    ],
)
def test_local_monitor_registry_does_not_merge_ranks(
    metric_id: int, name: str, family: Family, value_type: ValueType,
) -> None:
    metric = metric_by_name(name)
    assert (metric.metric_id, metric.family, metric.value_type) == (metric_id, family, value_type)
    assert metric.node_merge is NodeMerge.PER_RANK
    assert not metric.summarize_experts and not metric.job_identical and not metric.detailed
    assert metric.bins is None


def test_schema_version_is_one() -> None:
    assert SCHEMA_VERSION == 1


def test_metric_ids_are_unique() -> None:
    ids = [metric.metric_id for metric in METRICS]
    assert len(ids) == len(set(ids))


def test_metric_names_are_unique() -> None:
    names = [metric.name for metric in METRICS]
    assert len(names) == len(set(names))


def test_the_activation_points_are_those_the_catalog_has_activation_metrics_for() -> None:
    catalog_points = [metric.name.split("/")[1] for metric in METRICS if metric.name.endswith("/moments")]

    assert sorted(catalog_points) == sorted(LAYER_ACTIVATION_POINTS + STANDALONE_ACTIVATION_POINTS)


def test_every_metric_has_a_non_empty_description() -> None:
    for metric in METRICS:
        assert metric.description


def test_every_expert_count_holds_integer_words() -> None:
    # The node writes an expert count's INT32 words as its uint32 values without converting them.
    assert {metric.value_type for metric in METRICS if metric.family is Family.EXPERT_COUNTS} == {ValueType.INT32}


def test_metric_by_name_returns_the_matching_spec() -> None:
    metric = metric_by_name("router/tokens_per_expert")
    assert metric.metric_id == 1
    assert metric.family is Family.EXPERT_COUNTS
    assert metric.value_type is ValueType.INT32
    assert metric.node_merge is NodeMerge.PER_RANK

    stream_metric = metric_by_name("router/stream_tokens_per_expert")
    assert stream_metric.metric_id == 187
    assert stream_metric.family is Family.EXPERT_COUNTS
    assert stream_metric.value_type is ValueType.INT32
    assert stream_metric.node_merge is NodeMerge.PER_RANK


def test_metric_by_name_raises_key_error_for_unknown_metric() -> None:
    with pytest.raises(KeyError):
        metric_by_name("does/not_exist")


def test_metric_by_id_returns_every_metric_by_its_id() -> None:
    assert [metric_by_id(metric.metric_id) for metric in METRICS] == list(METRICS)


def test_metric_by_id_raises_key_error_for_a_retired_id() -> None:
    with pytest.raises(KeyError):
        metric_by_id(11)


def test_expert_bias_before_update_is_identical_merge() -> None:
    metric = metric_by_name("router/expert_bias_before_update")
    assert metric.family is Family.ROUTER_VECTORS
    assert metric.value_type is ValueType.FLOAT32
    assert metric.node_merge is NodeMerge.IDENTICAL
    assert metric.job_identical is True


def test_loss_total_is_a_per_rank_scalar() -> None:
    metric = metric_by_name("loss/total")
    assert metric.family is Family.SCALARS
    assert metric.value_type is ValueType.FLOAT32
    assert metric.node_merge is NodeMerge.PER_RANK


@pytest.mark.parametrize(
    ("name", "metric_id"),
    [("loss/hidden_z_loss_mla_kv_latent", 112), ("loss/hidden_z_loss_mtp0_mla_kv_latent", 113)],
)
def test_mla_latent_loss_metrics_preserve_rank_and_are_optional_when_disabled(name: str, metric_id: int) -> None:
    metric = metric_by_name(name)
    assert metric.metric_id == metric_id
    assert (metric.family, metric.value_type, metric.node_merge) == (
        Family.SCALARS, ValueType.FLOAT32, NodeMerge.PER_RANK,
    )
    assert metric.required is False
    assert metric.detailed is False
    assert metric.job_identical is False
    assert metric.bins is None
    assert "microbatch" in metric.description
    assert "Summed over enabled" in metric.description


def test_tokens_seen_is_deliberately_excluded_from_the_registry() -> None:
    with pytest.raises(KeyError):
        metric_by_name("tokens_seen")


def test_grad_local_norm_id_is_retired() -> None:
    ids = {metric.metric_id for metric in METRICS}
    assert 11 not in ids
    with pytest.raises(KeyError):
        metric_by_name("grad/local_norm")


@pytest.mark.parametrize(
    ("metric_id", "name"),
    [(13, "time/forward_gpu_s"), (14, "time/backward_gpu_s"), (15, "time/optimizer_gpu_s")],
)
def test_gpu_timing_metrics_are_per_rank_scalars_timed_with_cuda_events(metric_id: int, name: str) -> None:
    metric = metric_by_name(name)
    assert metric.metric_id == metric_id
    assert metric.family is Family.SCALARS
    assert metric.value_type is ValueType.FLOAT32
    assert metric.node_merge is NodeMerge.PER_RANK
    assert "CUDA events" in metric.description
    with pytest.raises(KeyError):
        metric_by_name(name.replace("_gpu_s", "_s"))


@pytest.mark.parametrize("name", ["time/forward_s", "time/backward_s", "time/optimizer_s"])
def test_wall_clock_timing_names_are_not_in_the_catalog(name: str) -> None:
    with pytest.raises(KeyError):
        metric_by_name(name)


def test_default_metric_spec_flags_are_off() -> None:
    metric = metric_by_name("loss/total")
    assert metric.bins is None
    assert metric.summarize_experts is False
    assert metric.detailed is False
    assert metric.job_identical is False


@pytest.mark.parametrize(
    "name",
    [
        "router/tokens_per_expert",
        "router/local_tokens_before_bias_update",
        "dispatcher/tokens_per_local_expert",
        "dispatcher/tokens_per_local_expert_no_pad",
    ],
)
def test_summarize_experts_is_set_on_exactly_the_dispatcher_and_router_count_metrics(name: str) -> None:
    assert metric_by_name(name).summarize_experts is True


def test_summarize_experts_is_off_for_every_other_metric() -> None:
    flagged = {metric.name for metric in METRICS if metric.summarize_experts}
    assert flagged == {
        "dispatcher/assignments_before_capacity",
        "dispatcher/valid_assignments_before_capacity",
        "dispatcher/assignments_after_capacity",
        "dispatcher/valid_assignments_after_capacity",
        "router/tokens_per_expert",
        "router/local_tokens_before_bias_update",
        "dispatcher/tokens_per_local_expert",
        "dispatcher/tokens_per_local_expert_no_pad",
    }


def test_histogram_bins_are_a_frozen_linear_grid() -> None:
    bins = HistogramBins(start=0.0, width=0.5, count=4)
    assert (bins.start, bins.width, bins.count) == (0.0, 0.5, 4)
    with pytest.raises(AttributeError):
        bins.count = 8


def test_token_loss_bins_constant() -> None:
    assert (TOKEN_LOSS_BINS.start, TOKEN_LOSS_BINS.width, TOKEN_LOSS_BINS.count) == (0.0, 0.125, 256)


def test_expert_loss_bins_constant() -> None:
    assert (EXPERT_LOSS_BINS.start, EXPERT_LOSS_BINS.width, EXPERT_LOSS_BINS.count) == (0.0, 0.25, 64)


def test_router_entropy_bins_constant() -> None:
    assert (ROUTER_ENTROPY_BINS.start, ROUTER_ENTROPY_BINS.width, ROUTER_ENTROPY_BINS.count) == (0.0, 1 / 32, 32)


@pytest.mark.parametrize(
    ("name", "bins"),
    [
        ("loss/token_hist_by_stream", TOKEN_LOSS_BINS),
        ("loss/mtp_token_hist", TOKEN_LOSS_BINS),
        ("expert_loss/hist", EXPERT_LOSS_BINS),
        ("router/entropy_hist", ROUTER_ENTROPY_BINS),
    ],
)
def test_histogram_metrics_carry_the_expected_bins(name: str, bins: HistogramBins) -> None:
    assert metric_by_name(name).bins is bins


@pytest.mark.parametrize(
    "name",
    ["expert_loss/hist", "expert_loss/stream_count", "expert_loss/stream_sum"],
)
def test_detailed_only_metrics_are_flagged(name: str) -> None:
    assert metric_by_name(name).detailed is True


def test_router_entropy_hist_is_built_every_step() -> None:
    assert metric_by_name("router/entropy_hist").detailed is False


def test_expert_loss_count_is_deliberately_not_summarized() -> None:
    assert metric_by_name("expert_loss/count").summarize_experts is False


def test_grad_clip_total_norm_is_identical_merge() -> None:
    metric = metric_by_name("grad/clip_total_norm")
    assert metric.node_merge is NodeMerge.IDENTICAL
    assert metric.family is Family.SCALARS
    assert metric.job_identical is True


def test_job_identical_is_set_on_exactly_bias_clip_norm_and_decay_parameters() -> None:
    flagged = {metric.name for metric in METRICS if metric.job_identical}
    assert flagged == {"router/expert_bias_before_update", "grad/clip_total_norm", "attn/linear/decay_params"}


def test_job_identical_metrics_always_use_identical_node_merge() -> None:
    for metric in METRICS:
        if metric.job_identical:
            assert metric.node_merge is NodeMerge.IDENTICAL


@pytest.mark.parametrize("name", ["grad/pre_reduce_sq_dense", "grad/pre_reduce_sq_expert"])
def test_pre_reduce_grad_metrics_are_per_rank_scalars_keyed_by_layer(name: str) -> None:
    metric = metric_by_name(name)
    assert metric.family is Family.SCALARS
    assert metric.node_merge is NodeMerge.PER_RANK
    assert "reduce-scatter" in metric.description


@pytest.mark.parametrize(
    "name", ["memory/allocated_gb", "memory/peak_allocated_gb", "memory/reserved_gb", "memory/alloc_retries"]
)
def test_memory_metrics_are_per_rank_host_scalars(name: str) -> None:
    metric = metric_by_name(name)
    assert metric.family is Family.SCALARS
    assert metric.value_type is ValueType.FLOAT32
    assert metric.node_merge is NodeMerge.PER_RANK


def test_loss_nonfinite_tokens_is_an_int_scalar() -> None:
    metric = metric_by_name("loss/nonfinite_tokens")
    assert metric.value_type is ValueType.INT32
    assert metric.node_merge is NodeMerge.PER_RANK


def test_loss_token_max_documents_the_zeroed_sentinel() -> None:
    description = metric_by_name("loss/token_max").description
    assert "0 when the microbatch has no such token" in description
    assert "-inf" not in description


@pytest.mark.parametrize(
    "name",
    [
        "loss/hidden_z_loss_attn",
        "loss/hidden_z_loss_moe",
        "loss/hidden_z_loss_dense_mlp",
    ],
)
def test_hidden_z_loss_metrics_are_per_rank_scalars(name: str) -> None:
    metric = metric_by_name(name)
    assert metric.family is Family.SCALARS
    assert metric.value_type is ValueType.FLOAT32
    assert metric.node_merge is NodeMerge.PER_RANK


def test_sampler_microbatches_is_a_per_rank_scalar() -> None:
    metric = metric_by_name("sampler/microbatches")
    assert metric.family is Family.SCALARS
    assert metric.value_type is ValueType.FLOAT32
    assert metric.node_merge is NodeMerge.PER_RANK
    assert "zeroed" in metric.description


def test_sampler_dropped_steps_describes_cumulative_semantics() -> None:
    metric = metric_by_name("sampler/dropped_steps")
    assert metric.family is Family.SCALARS
    assert metric.value_type is ValueType.FLOAT32
    assert metric.node_merge is NodeMerge.PER_RANK
    assert "cumulative" in metric.description.lower()
    assert "dropped_steps_per_rank" in metric.description


def test_activation_log2_bins_constant() -> None:
    assert (ACTIVATION_LOG2_BINS.start, ACTIVATION_LOG2_BINS.width, ACTIVATION_LOG2_BINS.count) == (
        -40.0,
        1.0,
        64,
    )


_EVERY_STEP_ACTIVATION_POINTS = ("residual", "attn_out", "mlp_out", "final_norm")
_DETAILED_ACTIVATION_POINTS = (
    "shared_expert_out",
    "attn_residual_contribution",
    "mlp_residual_contribution",
    "swiglu_in",
    "down_in",
    "embed",
)


@pytest.mark.parametrize("point", _EVERY_STEP_ACTIVATION_POINTS)
def test_every_step_activation_points_have_three_non_detailed_metrics(point: str) -> None:
    for suffix in ("moments", "token_rms_hist", "amax"):
        assert metric_by_name(f"act/{point}/{suffix}").detailed is False


@pytest.mark.parametrize("point", _DETAILED_ACTIVATION_POINTS)
def test_additional_activation_points_have_four_every_step_metrics(point: str) -> None:
    for suffix in ("moments", "exp_hist", "token_rms_hist", "amax"):
        assert metric_by_name(f"act/{point}/{suffix}").detailed is False


@pytest.mark.parametrize("point", _EVERY_STEP_ACTIVATION_POINTS + _DETAILED_ACTIVATION_POINTS)
def test_exp_hist_is_always_recorded_regardless_of_its_point(point: str) -> None:
    assert metric_by_name(f"act/{point}/exp_hist").detailed is False


@pytest.mark.parametrize("point", _EVERY_STEP_ACTIVATION_POINTS + _DETAILED_ACTIVATION_POINTS)
def test_activation_metrics_have_the_expected_families_bins_and_ids_in_order(point: str) -> None:
    moments = metric_by_name(f"act/{point}/moments")
    exp_hist = metric_by_name(f"act/{point}/exp_hist")
    rms_hist = metric_by_name(f"act/{point}/token_rms_hist")
    amax = metric_by_name(f"act/{point}/amax")

    assert (moments.family, moments.value_type, moments.node_merge) == (
        Family.ROUTER_VECTORS,
        ValueType.FLOAT32,
        NodeMerge.PER_RANK,
    )
    assert moments.bins is None

    for hist in (exp_hist, rms_hist):
        assert hist.family is Family.ROUTER_VECTORS
        assert hist.value_type is ValueType.FLOAT32
        assert hist.node_merge is NodeMerge.PER_RANK
        assert hist.bins is ACTIVATION_LOG2_BINS

    assert (amax.family, amax.value_type, amax.node_merge) == (
        Family.SCALARS,
        ValueType.FLOAT32,
        NodeMerge.PER_RANK,
    )
    assert amax.bins is None

    ids = (moments.metric_id, exp_hist.metric_id, rms_hist.metric_id, amax.metric_id)
    assert ids == (moments.metric_id, moments.metric_id + 1, moments.metric_id + 2, moments.metric_id + 3)


def test_activation_metric_ids_run_from_48_to_87_with_no_gaps() -> None:
    ids = sorted(
        metric_by_name(f"act/{point}/{suffix}").metric_id
        for point in _EVERY_STEP_ACTIVATION_POINTS + _DETAILED_ACTIVATION_POINTS
        for suffix in ("moments", "exp_hist", "token_rms_hist", "amax")
    )
    assert ids == list(range(48, 88))


@pytest.mark.parametrize(
    ("point", "metric_id"),
    [
        ("residual", 102), ("attn_out", 103), ("mlp_out", 104), ("final_norm", 105),
        ("shared_expert_out", 106), ("attn_residual_contribution", 107),
        ("mlp_residual_contribution", 108), ("swiglu_in", 109), ("down_in", 110), ("embed", 111),
    ],
)
def test_activation_counts_keep_exact_integer_storage_and_both_histogram_populations(
    point: str, metric_id: int,
) -> None:
    metric = metric_by_name(f"act/{point}/counts")
    assert metric.metric_id == metric_id
    assert (metric.family, metric.value_type, metric.node_merge) == (
        Family.EXPERT_COUNTS, ValueType.INT32, NodeMerge.PER_RANK,
    )
    assert ACTIVATION_COUNTS_WIDTH == 130
    assert metric.required is True
    assert metric.detailed is False
    assert metric.summarize_experts is False
    assert metric.job_identical is False
    assert metric.bins is None
    assert "[nonfinite_count, finite_count, 64 exponent bins, 64 token RMS bins]" in metric.description
    assert "2^31-1" in metric.description
    assert "Stored as UInt32 without conversion through float32" in metric.description


@pytest.mark.parametrize(
    ("point", "metric_id"),
    [
        ("residual", 161), ("attn_out", 162), ("mlp_out", 163), ("final_norm", 164),
        ("shared_expert_out", 165), ("attn_residual_contribution", 166),
        ("mlp_residual_contribution", 167), ("swiglu_in", 168), ("down_in", 169), ("embed", 170),
    ],
)
def test_activation_power_sums_are_float32_every_step_per_rank(point: str, metric_id: int) -> None:
    metric = metric_by_name(f"act/{point}/power_sums")
    assert (metric.metric_id, metric.family, metric.value_type, metric.node_merge) == (
        metric_id, Family.ROUTER_VECTORS, ValueType.FLOAT32, NodeMerge.PER_RANK,
    )
    assert (metric.required, metric.detailed, metric.bins) == (True, False, None)
    assert "[sum_cube, sum_fourth]" in metric.description
    assert f"act/{point}/moments" in metric.description


def test_activation_metric_descriptions_document_the_layer_convention() -> None:
    assert "num_layers + i" in metric_by_name("act/residual/moments").description
    assert "layer is always -1" in metric_by_name("act/final_norm/moments").description
    assert "layer is always -1" in metric_by_name("act/embed/moments").description


def test_moments_description_documents_the_four_values_and_counter_precision() -> None:
    description = metric_by_name("act/residual/moments").description
    assert "[sum, sumsq, nonfinite_count, finite_count]" in description
    assert "1e-6" in description


def test_exp_hist_description_documents_every_step() -> None:
    assert "every step" in metric_by_name("act/residual/exp_hist").description


@pytest.mark.parametrize("suffix", ["moments", "exp_hist", "token_rms_hist"])
def test_activation_counter_metrics_document_float32_precision(suffix: str) -> None:
    assert "1e-6" in metric_by_name(f"act/residual/{suffix}").description


def test_loss_position_buckets_constant() -> None:
    assert LOSS_POSITION_BUCKETS == 64


def test_sample_loss_bins_constant() -> None:
    assert (SAMPLE_LOSS_BINS.start, SAMPLE_LOSS_BINS.width, SAMPLE_LOSS_BINS.count) == (0.0, 0.125, 128)


@pytest.mark.parametrize(
    ("name", "metric_id", "family", "value_type"),
    [
        ("loss/position_sum", 89, Family.ROUTER_VECTORS, ValueType.FLOAT32),
        ("loss/position_count", 90, Family.EXPERT_COUNTS, ValueType.INT32),
        ("loss/sample_hist_by_stream", 91, Family.EXPERT_COUNTS, ValueType.INT32),
    ],
)
def test_new_loss_metrics_have_the_expected_ids_and_types(
    name: str, metric_id: int, family: Family, value_type: ValueType
) -> None:
    metric = metric_by_name(name)
    assert metric.metric_id == metric_id
    assert metric.family is family
    assert metric.value_type is value_type
    assert metric.node_merge is NodeMerge.PER_RANK


@pytest.mark.parametrize("name", ["loss/position_sum", "loss/position_count"])
def test_loss_position_metrics_are_built_every_step(name: str) -> None:
    metric = metric_by_name(name)
    assert metric.detailed is False
    assert metric.bins is None


def test_loss_position_sum_description_documents_the_bucket_formula() -> None:
    assert f"{LOSS_POSITION_BUCKETS} // seq_len" in metric_by_name("loss/position_sum").description


def test_loss_sample_hist_by_stream_is_detailed_with_sample_loss_bins() -> None:
    metric = metric_by_name("loss/sample_hist_by_stream")
    assert metric.detailed is True
    assert metric.bins is SAMPLE_LOSS_BINS


@pytest.mark.parametrize(
    ("metric_id", "name"),
    [
        (95, "dispatcher/assignments_before_capacity"),
        (96, "dispatcher/valid_assignments_before_capacity"),
        (97, "dispatcher/assignments_after_capacity"),
        (98, "dispatcher/valid_assignments_after_capacity"),
    ],
)
def test_dispatcher_snapshots_are_required_per_rank_counts(metric_id: int, name: str) -> None:
    metric = metric_by_name(name)
    assert (metric.metric_id, metric.family, metric.value_type, metric.node_merge,
            metric.required, metric.detailed, metric.summarize_experts) == (
        metric_id, Family.EXPERT_COUNTS, ValueType.INT32, NodeMerge.PER_RANK, True, False, True,
    )


def test_required_metrics_are_raw_every_microbatch_observations() -> None:
    required = {metric.name for metric in METRICS if metric.required}
    expected = {f"act/{point}/{suffix}" for point in _EVERY_STEP_ACTIVATION_POINTS + _DETAILED_ACTIVATION_POINTS
                for suffix in ("moments", "exp_hist", "token_rms_hist", "amax", "counts", "quantiles",
                               "quantile_counts", "power_sums")}
    expected.update({"router/tokens_per_expert", "router/stream_tokens_per_expert",
                     "dispatcher/assignments_before_capacity",
                     "dispatcher/valid_assignments_before_capacity", "dispatcher/assignments_after_capacity",
                     "dispatcher/valid_assignments_after_capacity",
                     "loss/token_quantiles", "loss/token_quantile_counts",
                     "loss/mtp_token_quantiles", "loss/mtp_token_quantile_counts",
                     "router/entropy_quantiles", "router/entropy_quantile_counts",
                     "attn/softmax/entropy_quantiles", "attn/softmax/entropy_quantile_counts"})
    assert required == expected
    assert all(not metric.detailed for metric in METRICS if metric.required)
    metric = metric_by_name("sampler/missing_slots")
    assert (metric.metric_id, metric.family, metric.node_merge, metric.required) == (
        99, Family.SCALARS, NodeMerge.PER_RANK, False,
    )


@pytest.mark.parametrize(("metric_id", "name", "family", "value_type"), [
    (150, "monitor/router_balance_local", Family.EXPERT_COUNTS, ValueType.INT32),
    (151, "monitor/expert_activation_sumsq_local", Family.EXPERT_COUNTS, ValueType.INT32),
    (152, "monitor/expert_activation_count_local", Family.EXPERT_COUNTS, ValueType.INT32),
    (153, "monitor/router_bias_local", Family.ROUTER_VECTORS, ValueType.FLOAT32),
    (154, "monitor/router_balance_counts_local", Family.EXPERT_COUNTS, ValueType.INT32),
    (156, "monitor/lr_group_local", Family.EXPERT_COUNTS, ValueType.INT32),
    (157, "monitor/memory_local", Family.EXPERT_COUNTS, ValueType.INT32),
    (158, "monitor/training_local", Family.EXPERT_COUNTS, ValueType.INT32),
    (159, "monitor/stage_time_local", Family.EXPERT_COUNTS, ValueType.INT32),
    (160, "monitor/materialized_weight_local", Family.ROUTER_VECTORS, ValueType.FLOAT32),
])
def test_monitor_payloads_retain_rank_identity_without_word_reductions(
    metric_id: int, name: str, family: Family, value_type: ValueType,
) -> None:
    metric = metric_by_name(name)
    assert (metric.metric_id, metric.family, metric.value_type, metric.node_merge,
            metric.job_identical, metric.summarize_experts, metric.detailed) == (
        metric_id, family, value_type, NodeMerge.PER_RANK, False, False, False,
    )


@pytest.mark.parametrize(("metric_id", "name", "fields"), [
    (171, "attn/softmax/lse_stats", "[rows, lse_sum x H, lse_min x H, lse_max x H], width=1+3H"),
    (172, "attn/softmax/qkv_stats",
     "[rows, q_nope_norm_sum x H, q_rope_norm_sum x H, k_nope_norm_sum x H, k_rope_norm_sum x H, "
     "v_norm_sum x H, q_amax x H, k_amax x H, v_amax x H], width=1+8H"),
    (173, "attn/softmax/gate_stats", "[rows, gate_sum x H, gate_min x H, gate_max x H], width=1+3H"),
    (179, "attn/linear/gate_stats",
     "[rows, beta_sum x Hv, beta_min x Hv, beta_max x Hv, alpha_sum x Hv, alpha_min x Hv, alpha_max x Hv, "
     "g_sum x Hv], width=1+7Hv"),
    (180, "attn/linear/norm_stats", "[rows, v_norm_sum x Hv, o_norm_sum x Hv, o_amax x Hv], width=1+3Hv"),
])
def test_attention_statistics_are_every_step_float32_vectors_of_each_rank(
    metric_id: int, name: str, fields: str,
) -> None:
    metric = metric_by_name(name)
    assert (metric.metric_id, metric.family, metric.value_type, metric.node_merge) == (
        metric_id, Family.ROUTER_VECTORS, ValueType.FLOAT32, NodeMerge.PER_RANK,
    )
    assert not (metric.required or metric.detailed or metric.job_identical or metric.collapse_equal_ranks)
    assert metric.bins is None and metric.alias_of is None and not metric.summarize_experts
    assert fields in metric.description


def test_attention_decay_parameters_are_stored_once_per_job() -> None:
    metric = metric_by_name("attn/linear/decay_params")
    assert (metric.metric_id, metric.family, metric.value_type, metric.node_merge, metric.job_identical) == (
        182, Family.ROUTER_VECTORS, ValueType.FLOAT32, NodeMerge.IDENTICAL, True,
    )
    assert not (metric.required or metric.detailed or metric.collapse_equal_ranks)
    assert "[A x Hv, softplus(dt_bias) x Hv], width=2Hv" in metric.description


def test_aliases_name_a_per_rank_metric_of_the_same_storage_that_is_not_an_alias_itself() -> None:
    aliases = {metric.name: metric.alias_of for metric in METRICS if metric.alias_of is not None}
    assert aliases == {
        "dispatcher/assignments_before_capacity": "router/tokens_per_expert",
        "dispatcher/assignments_after_capacity": "router/tokens_per_expert",
        "dispatcher/valid_assignments_after_capacity": "dispatcher/valid_assignments_before_capacity",
        "dispatcher/tokens_per_local_expert_no_pad": "dispatcher/tokens_per_local_expert",
    }
    for name, target_name in aliases.items():
        metric, target = metric_by_name(name), metric_by_name(target_name)
        assert (metric.family, metric.value_type) == (target.family, target.value_type) == (
            Family.EXPERT_COUNTS, ValueType.INT32,
        )
        assert metric.node_merge is target.node_merge is NodeMerge.PER_RANK
        assert target.alias_of is None

