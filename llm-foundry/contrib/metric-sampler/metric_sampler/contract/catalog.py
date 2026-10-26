from __future__ import annotations

from metric_sampler.contract.metrics import (
    ACTIVATION_LOG2_BINS,
    EXPERT_LOSS_BINS,
    LOSS_POSITION_BUCKETS,
    QUANTILE_LEVELS,
    ROUTER_ENTROPY_BINS,
    SAMPLE_LOSS_BINS,
    TOKEN_LOSS_BINS,
    Family,
    MetricSpec,
    NodeMerge,
    ValueType,
)

_COUNTER_PRECISION_NOTE = (
    "counters are accumulated as float32, exact up to 2^24 and within a relative error of "
    "about 1e-6 above that, the same accuracy any float32 running sum has; they are not exact "
    "integers on the largest models."
)


def _activation_point_metrics(
    first_metric_id: int,
    point: str,
    tensor_description: str,
    counts_metric_id: int,
    power_sums_metric_id: int,
) -> tuple[MetricSpec, ...]:
    """The moments, exponent histogram, token RMS histogram and maximum of an activation point at four
    consecutive ids from `first_metric_id`, then its exact counts and its power sums at their own ids."""
    return (
        MetricSpec(
            metric_id=first_metric_id,
            name=f"act/{point}/moments",
            family=Family.ROUTER_VECTORS,
            value_type=ValueType.FLOAT32,
            node_merge=NodeMerge.PER_RANK,
            unit="value",
            description=f"{tensor_description} Sum, sum of squares (both fp32, finite elements only), a "
            "count of non-finite (NaN or Inf) elements and a count of finite elements, laid out "
            f"as [sum, sumsq, nonfinite_count, finite_count]; the two {_COUNTER_PRECISION_NOTE} "
            "Recorded separately for each rank and microbatch.",
            required=True,
        ),
        MetricSpec(
            metric_id=first_metric_id + 1,
            name=f"act/{point}/exp_hist",
            family=Family.ROUTER_VECTORS,
            value_type=ValueType.FLOAT32,
            node_merge=NodeMerge.PER_RANK,
            unit="elements",
            description=f"{tensor_description} Histogram of floor(log2|x|) over finite elements; "
            f"{_COUNTER_PRECISION_NOTE} Recorded separately for each rank and microbatch on every step.",
            bins=ACTIVATION_LOG2_BINS,
            required=True,
        ),
        MetricSpec(
            metric_id=first_metric_id + 2,
            name=f"act/{point}/token_rms_hist",
            family=Family.ROUTER_VECTORS,
            value_type=ValueType.FLOAT32,
            node_merge=NodeMerge.PER_RANK,
            unit="tokens",
            description=f"{tensor_description} Histogram of floor(log2(rms)) of each row's RMS over the "
            "last axis (one row per token); a row with no finite elements is skipped entirely, "
            f"otherwise its RMS is computed from its finite elements only; {_COUNTER_PRECISION_NOTE} "
            "Recorded separately for each rank and microbatch.",
            bins=ACTIVATION_LOG2_BINS,
            required=True,
        ),
        MetricSpec(
            metric_id=first_metric_id + 3,
            name=f"act/{point}/amax",
            family=Family.SCALARS,
            value_type=ValueType.FLOAT32,
            node_merge=NodeMerge.PER_RANK,
            unit="value",
            description=f"{tensor_description} Maximum absolute value over finite elements, taken over "
            "every call within the microbatch; 0 both when every observed value was "
            "exactly zero and when the observed tensor is empty. Missing observations have no row.",
            required=True,
        ),
        MetricSpec(
            metric_id=counts_metric_id,
            name=f"act/{point}/counts",
            family=Family.EXPERT_COUNTS,
            value_type=ValueType.INT32,
            node_merge=NodeMerge.PER_RANK,
            unit="counts",
            description=f"{tensor_description} Exact integer counters for each rank and microbatch on every step: "
            "[nonfinite_count, finite_count, 64 exponent bins, 64 token RMS bins]. "
            "Both histograms use floor(log2) bins starting at -40 with width 1; zeros and "
            "underflow enter bin 0, and bin 63 includes overflow. Exponent bins count finite "
            "elements; RMS bins count rows with at least one finite element. All observations "
            "of this point within the microbatch accumulate; their total element count must "
            "not exceed 2^31-1. Stored as UInt32 without conversion through float32.",
            required=True,
        ),
        MetricSpec(
            metric_id=power_sums_metric_id,
            name=f"act/{point}/power_sums",
            family=Family.ROUTER_VECTORS,
            value_type=ValueType.FLOAT32,
            node_merge=NodeMerge.PER_RANK,
            unit="value",
            description=f"{tensor_description} Sums of the cubes and fourth powers of the finite elements, "
            f"[sum_cube, sum_fourth], accumulated in fp32 in the same pass as act/{point}/moments; with "
            f"its sum and sumsq and the finite count of act/{point}/counts they give the skewness and "
            "kurtosis of every rank and microbatch on every step. Central moments from raw sums "
            "cancel, so their precision falls as |mean|/std grows. A cube or fourth power beyond "
            "the float32 range makes the sums non-finite.",
            required=True,
        ),
    )


def _quantile_metrics(
    first_metric_id: int,
    quantiles_name: str,
    counts_name: str,
    population_description: str,
    *,
    float64: bool = False,
) -> tuple[MetricSpec, MetricSpec]:
    """The exact quantiles of a population and, at the next id, the counts of its finite, NaN, infinite and masked
    values; a float64 population is stored as pairs of UInt32 words."""
    stored_values = (
        "Nine Float64 values are stored losslessly as low/high UInt32 pairs, width=18. "
        if float64
        else "BF16/FP16/FP32 selected values are stored exactly as nine Float32 values. "
    )
    return (
        MetricSpec(
            metric_id=first_metric_id,
            name=quantiles_name,
            family=Family.EXPERT_COUNTS if float64 else Family.ROUTER_VECTORS,
            value_type=ValueType.INT32 if float64 else ValueType.FLOAT32,
            node_merge=NodeMerge.PER_RANK,
            unit="value",
            description=f"{population_description} Exact empirical inverse-CDF quantiles at "
            f"q={QUANTILE_LEVELS}: sorted finite values at max(ceil(q*N)-1, 0), "
            "without interpolation, sampling or histogram estimation. A point is observed "
            "once per microbatch; a repeated observation is rejected. NaN/Inf/masked values are "
            "counted separately; an empty finite population yields NaN quantiles. Selection "
            f"uses the source dtype. {stored_values}"
            "Rank, layer and microbatch are retained; quantiles must not be averaged to "
            "represent a combined population.",
            required=True,
        ),
        MetricSpec(
            metric_id=first_metric_id + 1,
            name=counts_name,
            family=Family.EXPERT_COUNTS,
            value_type=ValueType.INT32,
            node_merge=NodeMerge.PER_RANK,
            unit="elements",
            description=f"{population_description} Exact UInt32 population counts paired with {quantiles_name}: "
            "[finite, NaN, positive infinity, negative infinity, masked]. The first four "
            "counts cover unmasked inputs; masked inputs are counted once regardless of "
            "value. All observations in the rank/layer/microbatch are included. Total "
            "population must fit signed Int32; exceeding capacity aborts capture.",
            required=True,
        ),
    )


# A metric's id is stored with every value, so ids are never reused or renumbered; a retired metric keeps its id
# out of use.
METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        metric_id=1,
        name="router/tokens_per_expert",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Rank-local original top-k assignments per expert for one microbatch, before "
        "dispatcher capacity filtering, padding removal, EP exchange, or monitor reductions.",
        summarize_experts=True,
        required=True,
    ),
    MetricSpec(
        metric_id=2,
        name="router/local_tokens_before_bias_update",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Rank-local per-expert token count read by the aux-loss-free bias update, "
        "taken before that update resets it (see the v1/v2 silent-zero bug).",
        summarize_experts=True,
    ),
    MetricSpec(
        metric_id=3,
        name="dispatcher/tokens_per_local_expert",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Legacy rank-local routed assignments per expert before capacity filtering, "
        "including input padding tokens, captured after the forward pass of one microbatch.",
        summarize_experts=True,
    ),
    MetricSpec(
        metric_id=4,
        name="dispatcher/tokens_per_local_expert_no_pad",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Legacy rank-local routed assignments per expert before capacity filtering, "
        "excluding input padding tokens, captured after the forward pass of one microbatch.",
        summarize_experts=True,
        alias_of="dispatcher/tokens_per_local_expert",
    ),
    MetricSpec(
        metric_id=5,
        name="router/scores_mean",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="score",
        description="Legacy rank-local mean router score per expert. Raw records retain each rank; "
        "means require token-count weighting when combining unequal microbatches.",
    ),
    MetricSpec(
        metric_id=6,
        name="router/expert_bias_before_update",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.IDENTICAL,
        unit="bias",
        description="Already-reduced expert bias vector, identical across ranks of a node by "
        "construction, taken before UpdateBias applies this step's adjustment.",
        job_identical=True,
    ),
    MetricSpec(
        metric_id=7,
        name="loss/total",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Rank-local total loss for the step, taken after the last microbatch's backward pass.",
    ),
    MetricSpec(
        metric_id=8,
        name="loss/ce",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Rank-local cross-entropy loss component for the step, taken after the "
        "last microbatch's backward pass.",
    ),
    MetricSpec(
        metric_id=9,
        name="loss/mtp",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Rank-local multi-token-prediction loss component for the step, taken "
        "after the last microbatch's backward pass.",
    ),
    MetricSpec(
        metric_id=10,
        name="loss/z_loss",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Rank-local z-loss component for the step, taken after the last microbatch's backward pass.",
    ),
    # Id 11 (grad/local_norm) is retired: it measured the already-reduced gradient, not a per-rank quantity.
    MetricSpec(
        metric_id=12,
        name="time/step_s",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="s",
        description="Rank-local wall-clock duration of the whole step, taken at step end.",
    ),
    MetricSpec(
        metric_id=13,
        name="time/forward_gpu_s",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="s",
        description="Rank-local duration of one microbatch's forward pass on the GPU timeline "
        "(the default stream, including communication waits), timed with CUDA events and "
        "computed after the step ends without synchronizing the training thread.",
    ),
    MetricSpec(
        metric_id=14,
        name="time/backward_gpu_s",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="s",
        description="Rank-local duration of one microbatch's backward pass on the GPU timeline "
        "(the default stream, including communication waits), timed with CUDA events and "
        "computed after the step ends without synchronizing the training thread.",
    ),
    MetricSpec(
        metric_id=15,
        name="time/optimizer_gpu_s",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="s",
        description="Rank-local duration of the step's optimizer update on the GPU timeline "
        "(the default stream, including communication waits), timed with CUDA events and "
        "computed after the step ends without synchronizing the training thread.",
    ),
    MetricSpec(
        metric_id=16,
        name="lr",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="lr",
        description="Learning rate applied to this step's optimizer update, identical on every "
        "rank, taken before the optimizer step.",
    ),
    MetricSpec(
        metric_id=17,
        name="sampler/dropped_steps",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="steps",
        description="Cumulative number of steps this rank dropped since the capture started "
        "because no ring slot was free; per-step drops are in the node manifest "
        "(dropped_steps_per_rank).",
    ),
    MetricSpec(
        metric_id=18,
        name="sampler/step_end_s",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="s",
        description="Rank-local wall-clock cost of the recorder's own step_end call, taken "
        "immediately after it returns.",
    ),
    MetricSpec(
        metric_id=19,
        name="summary/dead_experts",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="experts",
        description="Derived by the node aggregator, not recorded by ranks: count of zero "
        "entries in a rank's own per-expert-count array for a summed EXPERT_COUNTS key.",
    ),
    MetricSpec(
        metric_id=20,
        name="summary/max_expert_share",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="ratio",
        description="Derived by the node aggregator, not recorded by ranks: a rank's busiest "
        "expert's share (max / sum, 0 when the sum is 0) of a summed EXPERT_COUNTS key.",
    ),
    MetricSpec(
        metric_id=21,
        name="loss/hidden_z_loss_attn",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Rank-local hidden-state z-loss from the attention blocks, taken after "
        "each microbatch's backward pass.",
    ),
    MetricSpec(
        metric_id=22,
        name="loss/hidden_z_loss_moe",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Rank-local hidden-state z-loss from the MoE blocks, taken after each microbatch's backward pass.",
    ),
    MetricSpec(
        metric_id=23,
        name="loss/hidden_z_loss_dense_mlp",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Rank-local hidden-state z-loss from the dense MLP blocks, taken after "
        "each microbatch's backward pass.",
    ),
    MetricSpec(
        metric_id=24,
        name="sampler/microbatches",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="microbatches",
        description="Number of microbatches the rank recorded in this step; slots for higher "
        "microbatch indices were zeroed and carry no data.",
    ),
    MetricSpec(
        metric_id=25,
        name="loss/token_hist_by_stream",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Histogram of the main segment's per-token cross-entropy (z-loss and "
        "hidden z-loss are not included), one row of bins per data stream, "
        "values laid out as [stream][bin] (width streams * 256); counts only tokens with a "
        "valid label and a finite loss, recorded separately for each microbatch.",
        bins=TOKEN_LOSS_BINS,
    ),
    MetricSpec(
        metric_id=26,
        name="loss/mtp_token_hist",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Histogram of an MTP head's per-token loss, keyed by that head's MTP "
        "layer index; same token-validity rule as loss/token_hist_by_stream but pooled across "
        "streams (width 256), recorded separately for each microbatch.",
        bins=TOKEN_LOSS_BINS,
    ),
    MetricSpec(
        metric_id=27,
        name="loss/stream_sum",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Sum of the main segment's per-token loss by stream, over tokens with a "
        "valid label and a finite loss, recorded separately for each microbatch.",
    ),
    MetricSpec(
        metric_id=28,
        name="loss/stream_sumsq",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats^2",
        description="Sum of squares of the main segment's per-token loss by stream, over "
        "tokens with a valid label and a finite loss, recorded separately for each "
        "microbatch.",
    ),
    MetricSpec(
        metric_id=29,
        name="loss/stream_count",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Count of the main segment's tokens with a valid label and a finite loss, "
        "by stream, recorded separately for each microbatch; the token count backing "
        "loss/stream_sum and loss/stream_sumsq.",
    ),
    MetricSpec(
        metric_id=30,
        name="loss/stream_samples",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="samples",
        description="Count of samples (sequences) by stream, recorded separately for each "
        "microbatch, regardless of how many of their tokens are valid; distinguishes a "
        "genuinely bad stream from one made of short documents.",
    ),
    MetricSpec(
        metric_id=31,
        name="loss/token_max",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Maximum per-token loss of the main segment among tokens with a valid "
        "label and a finite loss, within one microbatch; 0 when the microbatch has no such "
        "token (an atomic max onto a zeroed slot; cross-entropy is never negative, so this "
        "cannot be confused with a genuine maximum).",
    ),
    MetricSpec(
        metric_id=32,
        name="loss/nonfinite_tokens",
        family=Family.SCALARS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Count of tokens with a valid label but a non-finite (NaN or Inf) loss, "
        "across every segment (the main segment and every MTP head), within one microbatch.",
    ),
    MetricSpec(
        metric_id=33,
        name="expert_loss/sum",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Sum of loss attributed to each expert of a routed layer by the router's "
        "top-k assignment, before capacity dropping; the loss is that of the segment the "
        "layer belongs to (the main segment for a main layer, head h for MTP layer h), "
        "recorded separately for each microbatch.",
    ),
    MetricSpec(
        metric_id=34,
        name="expert_loss/sumsq",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats^2",
        description="Sum of squares of loss attributed to each expert of a routed layer by "
        "the router's top-k assignment, before capacity dropping; same segment rule as "
        "expert_loss/sum, recorded separately for each microbatch.",
    ),
    MetricSpec(
        metric_id=35,
        name="expert_loss/count",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Count of tokens routed to each expert of a routed layer by the router's "
        "top-k assignment, before capacity dropping, recorded separately for each "
        "microbatch; deliberately not summarized (dead experts and busiest-expert share are "
        "already covered by router/tokens_per_expert).",
    ),
    MetricSpec(
        metric_id=36,
        name="expert_loss/hist",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Per-expert histogram of the loss from expert_loss/sum's assignment, "
        "values laid out as [expert][bin]; detailed-mode only, recorded separately for each "
        "microbatch.",
        bins=EXPERT_LOSS_BINS,
        detailed=True,
    ),
    MetricSpec(
        metric_id=37,
        name="expert_loss/stream_count",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Per-expert, per-stream token count from expert_loss/sum's assignment, "
        "values laid out as [expert][stream]; an MTP layer uses the sample's own stream; "
        "detailed-mode only, recorded separately for each microbatch.",
        detailed=True,
    ),
    MetricSpec(
        metric_id=38,
        name="expert_loss/stream_sum",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Per-expert, per-stream loss sum from expert_loss/sum's assignment, "
        "values laid out as [expert][stream]; an MTP layer uses the sample's own stream; "
        "detailed-mode only, recorded separately for each microbatch.",
        detailed=True,
    ),
    MetricSpec(
        metric_id=39,
        name="router/entropy_hist",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Per-layer histogram of router entropy, computed from the router's scores "
        "normalized to sum to 1 across experts (before the bias term) and divided by "
        "log(experts), with FP64 normalization and logarithms; built over every token of the "
        "layer's segment, including label-invalid tokens. Recorded every step, separately "
        "for each rank and microbatch. Older capture versions sampled every eighth token; "
        "their source metadata determines the historical population.",
        bins=ROUTER_ENTROPY_BINS,
    ),
    MetricSpec(
        metric_id=40,
        name="grad/pre_reduce_sq_dense",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="norm^2",
        description="Sum of squares of this rank's local gradient for dense parameters, "
        "accumulated across the step's microbatches, taken before the inter-rank "
        "reduce-scatter; keyed by decoder layer index, or -1 for parameters outside a layer.",
    ),
    MetricSpec(
        metric_id=41,
        name="grad/pre_reduce_sq_expert",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="norm^2",
        description="Sum of squares of this rank's local gradient for expert parameters, "
        "accumulated across the step's microbatches, taken before the inter-rank "
        "reduce-scatter; keyed by decoder layer index, or -1 for parameters outside a layer.",
    ),
    MetricSpec(
        metric_id=42,
        name="grad/clip_total_norm",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.IDENTICAL,
        unit="norm",
        description="Global gradient norm computed by the clipping algorithm before it "
        "applies clipping; identical across every rank by construction.",
        job_identical=True,
    ),
    MetricSpec(
        metric_id=43,
        name="memory/allocated_gb",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="gb",
        description="Rank-local CUDA allocator bytes currently allocated, in GiB, taken from "
        "the host at the end of the step.",
    ),
    MetricSpec(
        metric_id=44,
        name="memory/peak_allocated_gb",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="gb",
        description="Rank-local peak CUDA allocator bytes allocated since the process started "
        "or since anyone last reset the peak counter, in GiB, taken from the host at the end "
        "of the step.",
    ),
    MetricSpec(
        metric_id=45,
        name="memory/reserved_gb",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="gb",
        description="Rank-local CUDA allocator bytes currently reserved, in GiB, taken from "
        "the host at the end of the step.",
    ),
    MetricSpec(
        metric_id=46,
        name="memory/alloc_retries",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="retries",
        description="Rank-local cumulative count of CUDA allocator retries since the process "
        "started, taken from the host at the end of the step.",
    ),
    MetricSpec(
        metric_id=47,
        name="time/step_gpu_s",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="s",
        description="Rank-local duration of the whole step on the GPU timeline, timed with "
        "CUDA events from batch_start to batch_end.",
    ),
    *_activation_point_metrics(
        first_metric_id=48,
        point="residual",
        tensor_description="Decoder layer output, i.e. the residual stream state after the layer's attention and "
        "MLP/MoE contributions have both been added; layer is the decoder layer index, or "
        "num_layers + i for MTP layer i.",
        counts_metric_id=102,
        power_sums_metric_id=161,
    ),
    *_activation_point_metrics(
        first_metric_id=52,
        point="attn_out",
        tensor_description="Attention block output before it is added to the residual stream; layer is the "
        "decoder layer index, or num_layers + i for MTP layer i.",
        counts_metric_id=103,
        power_sums_metric_id=162,
    ),
    *_activation_point_metrics(
        first_metric_id=56,
        point="mlp_out",
        tensor_description="MoE block output for a routed layer, or the dense MLP output for a dense layer, "
        "before it is added to the residual stream; layer is the decoder layer index, or "
        "num_layers + i for MTP layer i.",
        counts_metric_id=104,
        power_sums_metric_id=163,
    ),
    *_activation_point_metrics(
        first_metric_id=60,
        point="final_norm",
        tensor_description="Output of the model's final norm, i.e. the input to lm_head; not scoped to a layer, "
        "so layer is always -1.",
        counts_metric_id=105,
        power_sums_metric_id=164,
    ),
    *_activation_point_metrics(
        first_metric_id=64,
        point="shared_expert_out",
        tensor_description="Shared expert output before it is added to the routed experts' combined output; "
        "layer is the decoder layer index of the routed layer.",
        counts_metric_id=106,
        power_sums_metric_id=165,
    ),
    *_activation_point_metrics(
        first_metric_id=68,
        point="attn_residual_contribution",
        tensor_description="Attention block's contribution actually added to the residual stream, after any "
        "post-attention scaling; kept separate from act/attn_out to catch a scaling bug "
        "between the two; layer is the decoder layer index, or num_layers + i for MTP layer i.",
        counts_metric_id=107,
        power_sums_metric_id=166,
    ),
    *_activation_point_metrics(
        first_metric_id=72,
        point="mlp_residual_contribution",
        tensor_description="MoE or dense MLP block's contribution actually added to the residual stream, after "
        "any post-block scaling; kept separate from act/mlp_out to catch a scaling bug "
        "between the two; layer is the decoder layer index, or num_layers + i for MTP layer i.",
        counts_metric_id=108,
        power_sums_metric_id=167,
    ),
    *_activation_point_metrics(
        first_metric_id=76,
        point="swiglu_in",
        tensor_description="Expert SwiGLU input, the gate and up projections concatenated along the last axis "
        "(gate‖up); layer is the decoder layer index of the routed layer.",
        counts_metric_id=109,
        power_sums_metric_id=168,
    ),
    *_activation_point_metrics(
        first_metric_id=80,
        point="down_in",
        tensor_description="Expert down-projection input, i.e. the SwiGLU activation output; layer is the "
        "decoder layer index of the routed layer.",
        counts_metric_id=110,
        power_sums_metric_id=169,
    ),
    *_activation_point_metrics(
        first_metric_id=84,
        point="embed",
        tensor_description="Token embedding output, the input to the first decoder layer; not scoped to a layer, "
        "so layer is always -1.",
        counts_metric_id=111,
        power_sums_metric_id=170,
    ),
    MetricSpec(
        metric_id=88,
        name="time/loop_s",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="s",
        description="Rank-local wall-clock duration of the PREVIOUS step's whole training loop, "
        "from its batch_start to this step's batch_start, recorded on this step: unlike "
        "time/step_s it includes every callback and logger that runs after the sampler in "
        "batch_end (monitors, tensorboard), which can dominate the steps they fire on.",
    ),
    MetricSpec(
        metric_id=89,
        name="loss/position_sum",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Sum of the main segment's per-token loss by position bucket "
        f"(position * {LOSS_POSITION_BUCKETS} // seq_len; position within the packed "
        "sequence, not within a document), over tokens with a valid label and a finite "
        "loss, recorded separately for each microbatch.",
    ),
    MetricSpec(
        metric_id=90,
        name="loss/position_count",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Count of the main segment's tokens with a valid label and a finite "
        "loss, by the same position bucket as loss/position_sum, accumulated over the "
        "step's microbatches.",
    ),
    MetricSpec(
        metric_id=91,
        name="loss/sample_hist_by_stream",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="samples",
        description="Histogram of the main segment's per-sample mean loss (mean over that "
        "sample's tokens with a valid label and a finite loss; a sample with none is not "
        "counted), one row of bins per data stream, values laid out as [stream][bin] "
        f"(width streams * {SAMPLE_LOSS_BINS.count}); detailed-mode only, accumulated over "
        "each microbatch.",
        bins=SAMPLE_LOSS_BINS,
        detailed=True,
    ),
    MetricSpec(
        metric_id=92,
        name="sampler/dumps_skipped",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="dumps",
        description="Cumulative number of activation-dump attempts this rank skipped since dump "
        "capture started, because the dump buffer was still writing out a previous dump or the "
        "attempt's max_gb budget was reached; 0 while activation dumps are disabled.",
    ),
    MetricSpec(
        metric_id=93,
        name="loss/worst_sample_info",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="sample",
        description="Up to four samples per rank and microbatch on detailed steps, selected by "
        "non-finite labelled loss first, then descending finite-token mean loss. Each sample "
        "stores [stream_id, stream-local shard_id, index_in_shard, loss_position, finite_tokens, "
        "nonfinite_tokens] as uint32 words. Position is zero-based in the transformed loss "
        "sequence: first non-finite labelled token, otherwise the maximum finite loss. "
        "Zero token counts mark unused entries. Written only when dataset identities are available.",
        detailed=True,
    ),
    MetricSpec(
        metric_id=94,
        name="loss/worst_sample_stats",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="loss",
        description="Finite-token [mean_loss, max_loss] pairs for the same samples, in the same "
        "order, as loss/worst_sample_info. Entries with no finite labelled tokens have no "
        "meaningful loss value and must be filtered using the paired token counts.",
        detailed=True,
    ),
    MetricSpec(
        metric_id=95,
        name="dispatcher/assignments_before_capacity",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Rank-local assignments per expert before capacity filtering, including input padding tokens. "
        "Captured separately for each microbatch before EP exchange or monitor reductions.",
        summarize_experts=True,
        required=True,
        alias_of="router/tokens_per_expert",
    ),
    MetricSpec(
        metric_id=96,
        name="dispatcher/valid_assignments_before_capacity",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Rank-local assignments per expert before capacity filtering, excluding padding and "
        "prefix-masked tokens. "
        "Captured separately for each microbatch before EP exchange or monitor reductions.",
        summarize_experts=True,
        required=True,
    ),
    MetricSpec(
        metric_id=97,
        name="dispatcher/assignments_after_capacity",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Rank-local assignments per expert after capacity filtering, including input padding tokens. "
        "Captured separately for each microbatch before EP exchange or monitor reductions.",
        summarize_experts=True,
        required=True,
        alias_of="router/tokens_per_expert",
    ),
    MetricSpec(
        metric_id=98,
        name="dispatcher/valid_assignments_after_capacity",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Rank-local assignments per expert after capacity filtering, excluding padding and "
        "prefix-masked tokens. "
        "Captured separately for each microbatch before EP exchange or monitor reductions.",
        summarize_experts=True,
        required=True,
        alias_of="dispatcher/valid_assignments_before_capacity",
    ),
    MetricSpec(
        metric_id=99,
        name="sampler/missing_slots",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="slots",
        description="Number of required capture slots absent from this rank's recorded step. "
        "Computed by the node process from the declared layout and recorded mask.",
    ),
    MetricSpec(
        metric_id=100,
        name="sampler/ring_used_slots",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="slots",
        description="Occupied shared-memory slots before submitting this rank's completed step.",
    ),
    MetricSpec(
        metric_id=101,
        name="sampler/ring_capacity_slots",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="slots",
        description="Capacity of this rank's shared-memory ring.",
    ),
    MetricSpec(
        metric_id=112,
        name="loss/hidden_z_loss_mla_kv_latent",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Rank-local detached hidden z-loss on normalized and scaled shared MLA KV "
        "latents before their K/V projections. Summed over enabled trunk modules by the model, "
        "recorded separately for each microbatch before monitor reductions; not per-layer data.",
    ),
    MetricSpec(
        metric_id=113,
        name="loss/hidden_z_loss_mtp0_mla_kv_latent",
        family=Family.SCALARS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Rank-local detached hidden z-loss on normalized and scaled shared MLA KV "
        "latents in MTP predictor 0, before their K/V projections. Summed over enabled modules "
        "of this predictor by the model, recorded separately for each microbatch before monitor reductions.",
    ),
    *_quantile_metrics(
        first_metric_id=114,
        quantiles_name="act/residual/quantiles",
        counts_name="act/residual/quantile_counts",
        population_description="Decoder layer output, over all tensor elements.",
    ),
    *_quantile_metrics(
        first_metric_id=116,
        quantiles_name="act/attn_out/quantiles",
        counts_name="act/attn_out/quantile_counts",
        population_description="Attention output before residual addition, over all tensor elements.",
    ),
    *_quantile_metrics(
        first_metric_id=118,
        quantiles_name="act/mlp_out/quantiles",
        counts_name="act/mlp_out/quantile_counts",
        population_description="MLP/MoE output before residual addition, over all tensor elements.",
    ),
    *_quantile_metrics(
        first_metric_id=120,
        quantiles_name="act/final_norm/quantiles",
        counts_name="act/final_norm/quantile_counts",
        population_description="Final normalization output, over all tensor elements; layer=-1.",
    ),
    *_quantile_metrics(
        first_metric_id=122,
        quantiles_name="act/shared_expert_out/quantiles",
        counts_name="act/shared_expert_out/quantile_counts",
        population_description="Shared expert output, over all tensor elements.",
    ),
    *_quantile_metrics(
        first_metric_id=124,
        quantiles_name="act/attn_residual_contribution/quantiles",
        counts_name="act/attn_residual_contribution/quantile_counts",
        population_description="Scaled attention residual contribution, over all tensor elements.",
    ),
    *_quantile_metrics(
        first_metric_id=126,
        quantiles_name="act/mlp_residual_contribution/quantiles",
        counts_name="act/mlp_residual_contribution/quantile_counts",
        population_description="Scaled MLP/MoE residual contribution, over all tensor elements.",
    ),
    *_quantile_metrics(
        first_metric_id=128,
        quantiles_name="act/swiglu_in/quantiles",
        counts_name="act/swiglu_in/quantile_counts",
        population_description="Local expert gate/up projection outputs, over all dispatched tensor elements.",
    ),
    *_quantile_metrics(
        first_metric_id=130,
        quantiles_name="act/down_in/quantiles",
        counts_name="act/down_in/quantile_counts",
        population_description="Local expert down-projection inputs, over all dispatched tensor elements.",
    ),
    *_quantile_metrics(
        first_metric_id=132,
        quantiles_name="act/embed/quantiles",
        counts_name="act/embed/quantile_counts",
        population_description="Token embedding output, over all tensor elements; layer=-1.",
    ),
    *_quantile_metrics(
        first_metric_id=134,
        quantiles_name="loss/token_quantiles",
        counts_name="loss/token_quantile_counts",
        population_description="Main-head per-token loss; label-invalid positions are masked.",
    ),
    *_quantile_metrics(
        first_metric_id=136,
        quantiles_name="loss/mtp_token_quantiles",
        counts_name="loss/mtp_token_quantile_counts",
        population_description="MTP-head per-token loss, with layer identifying the head; "
        "label-invalid positions are masked.",
    ),
    *_quantile_metrics(
        first_metric_id=138,
        quantiles_name="router/entropy_quantiles",
        counts_name="router/entropy_quantile_counts",
        population_description="Per-token normalized router entropy, over every router input row.",
        float64=True,
    ),
    MetricSpec(
        metric_id=140,
        name="monitor/router_margin_local",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="margin",
        description="Rank/layer/microbatch router margins over the full token population, "
        "before monitor reductions: [score median gap, score median near, score p25, score p10, "
        "logit median gap, logit median near, logit p25, logit p10]. Percentiles use the "
        "ordinary monitor's linear interpolation, without its optional token subsampling.",
    ),
    MetricSpec(
        metric_id=141,
        name="monitor/optimizer_health_local",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="float64_bits",
        description="Local optimizer-shard sufficient statistics before monitor SUM reduction, "
        "11 named parameter groups by 6 Float64 accumulators. Each Float64 is stored losslessly "
        "as two UInt32 words, low word then high word; width=132. Never sum the encoded words. "
        "Group identities and accumulator fields are recorded in monitor metadata; layer=-1, microbatch=-1.",
    ),
    MetricSpec(
        metric_id=142,
        name="monitor/update_weight_local",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="float64_bits",
        description="Local parameter-shard [numel, update_abs_sum, weight_sum, weight_square_sum] "
        "before monitor SUM reduction; learning-rate scaling is already included in update_abs_sum. "
        "Four Float64 values stored losslessly as low/high UInt32 word pairs, width=8; never sum "
        "encoded words. Layer indexes the parameter FQN catalog; microbatch=-1.",
    ),
    MetricSpec(
        metric_id=143,
        name="monitor/topk_local_counts",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Rank-local [valid label tokens, correct top1, correct top5, correct top20] "
        "from available logits before monitor reductions, separately for each microbatch. "
        "Unavailable logits are not zero accuracy and must not produce fabricated counts.",
    ),
    MetricSpec(
        metric_id=144,
        name="monitor/activation_local",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="value",
        description="Ordinary activation monitor's local [amax, mean row L2 norm] before inter-rank "
        "reductions, retaining microbatch and the activation FQN catalog index as layer. "
        "The catalog distinguishes module input, output and gradient observations. Repeated "
        "calls form one population with a row-count-weighted mean of last-axis row L2 norms; "
        "use paired observation and row counts to distinguish absent values.",
    ),
    MetricSpec(
        metric_id=145,
        name="monitor/weight_local",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="value",
        description="Ordinary weight monitor's local parameter-shard [amax, l2norm] before "
        "inter-rank reductions. Layer indexes the parameter FQN catalog; microbatch=-1. "
        "A shard statistic is not a full-parameter statistic when the parameter is partitioned.",
    ),
    MetricSpec(
        metric_id=146,
        name="monitor/activation_local_observations",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="observations",
        description="Exact [observation_count, row_count] for floating activation or gradient "
        "observations represented by monitor/activation_local for the same rank/FQN/direction/microbatch. "
        "A zero observation count explicitly "
        "means no observation (including optional outputs and unused gradients), not zero "
        "activation. All repeated calls contribute; the paired statistics describe their union. "
        "Stored once per node (rank -1) when every rank of the node reports the same counts.",
        collapse_equal_ranks=True,
    ),
    MetricSpec(
        metric_id=147,
        name="monitor/training_local_counts",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="counts",
        description="Rank-local [skip_updates, total_updates, zeroed_module_grads, processed_module_grads] "
        "before TrainingMetricsMonitor reductions or counter reset. Four UInt64 counters stored as "
        "low/high UInt32 word pairs, followed by four 0/1 presence words; width=12. Absent producers "
        "have presence=0 and are not measured zeros. Layer=-1, microbatch=-1.",
    ),
    MetricSpec(
        metric_id=149,
        name="monitor/optimizer_local",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="value",
        description="OptimizerMonitor's local parameter-shard values before monitor reductions: "
        "[l2_grad, l2_moment, l2_moment_sq, l2_param, l2_update, l2_moment_bc, l2_moment_sq_bc, "
        "amax_moment, amax_moment_sq], followed by nine 0/1 presence values; width=18. "
        "Missing fields are explicit, not measured zeros. Layer indexes the parameter FQN catalog; "
        "microbatch=-1. Gradients here are optimizer inputs and may already have undergone FSDP reduction.",
    ),
    MetricSpec(
        metric_id=150,
        name="monitor/router_balance_local",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="float64_bits",
        description="Rank/layer/microbatch router statistics before monitor reductions: sums of "
        "[normalized top-k scores, raw top-k scores, biased top-k scores, ordinary biased-score "
        "entropy, router z-loss, sequence auxiliary loss, alpha]. Top-k sums include valid token "
        "rows; entropy includes all rows; loss and alpha values accumulate across forward calls. "
        "Use paired all-token, valid-token and call counts for the corresponding means. Seven "
        "Float64 values stored as low/high UInt32 word pairs, width=14; never sum encoded words.",
    ),
    MetricSpec(
        metric_id=151,
        name="monitor/expert_activation_sumsq_local",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="float64_bits",
        description="Per-expert squared activation norm sums before monitor communications, "
        "retaining rank/layer/microbatch. Float64 accumulation over every local dispatched row; "
        "each global expert occupies a low/high UInt32 word pair. Expert ownership is specified "
        "in the capture contract; zero outside the owned slice is not a remote measurement. "
        "Use exact paired token counts for means; never sum encoded words.",
    ),
    MetricSpec(
        metric_id=152,
        name="monitor/expert_activation_count_local",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Exact original integer token counts for each global expert's local activation "
        "population, before monitor accumulation or communication. Retains rank/layer/microbatch; "
        "all repeated calls contribute. Zero outside the explicitly owned expert slice is not a "
        "remote measurement. Counts never pass through the activation floating-point dtype.",
    ),
    MetricSpec(
        metric_id=153,
        name="monitor/router_bias_local",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="score",
        description="Complete rank-local router bias vector before monitor reductions, with global "
        "expert position, layer, step and microbatch preserved. Repeated observations in one "
        "microbatch must have identical bias values; a change is rejected rather than averaged. "
        "Stored once per node (rank -1) when every rank of the node reports the same vector.",
        collapse_equal_ranks=True,
    ),
    MetricSpec(
        metric_id=154,
        name="monitor/router_balance_counts_local",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="counts",
        description="Exact rank/layer/microbatch [all_tokens, valid_tokens, forward_calls] for "
        "monitor/router_balance_local and router/scores_mean. All batch and sequence rows and "
        "all repeated forward calls contribute; no first-sequence selection or subsampling.",
    ),
    MetricSpec(
        metric_id=156,
        name="monitor/lr_group_local",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="float64_bits",
        description="Rank-local learning rate for each optimizer parameter group, preserving the "
        "original Python Float64 value as low/high UInt32 words. Layer indexes the canonical "
        "optimizer/class/group catalog; microbatch=-1. Never numerically aggregate encoded words.",
    ),
    MetricSpec(
        metric_id=157,
        name="monitor/memory_local",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="counts",
        description="MemoryMonitor's rank-local integer allocator statistics and device free/total "
        "bytes at AFTER_TRAIN_BATCH, before optimizer execution and before display rounding. "
        "Nine allocator fields followed by free/total: eleven UInt64 low/high word pairs, then "
        "eleven presence words, width=33. Exact field order is in the capture contract. "
        "Absent fields are not measured zeros; this is not the later batch-end memory snapshot.",
    ),
    MetricSpec(
        metric_id=158,
        name="monitor/training_local",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="bits",
        description="Original local [samples, tokens, batch_seconds, device_FLOPs] before Trainer "
        "sample/token reductions, batch-time broadcast and SpeedMonitor FLOP reduction. Samples "
        "and tokens are UInt64, time and FLOPs Float64; each occupies low/high UInt32 words, "
        "followed by four presence words, width=12. Missing fields are explicit. These source "
        "values retain rank identity and can support local throughput calculations.",
    ),
    MetricSpec(
        metric_id=159,
        name="monitor/stage_time_local",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="seconds_bits",
        description="StageTimeProfiler's local wall-clock durations before reset: [batch_total, "
        "forward, loss_calculation, backward, optimizer_step, dataloader_fetch]. Six Float64 "
        "low/high word pairs followed by six presence words, width=18. These host wall-clock "
        "intervals are not CUDA event timings. Layer=-1, microbatch=-1.",
    ),
    MetricSpec(
        metric_id=160,
        name="monitor/materialized_weight_local",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="value",
        description="WeightMonitor's first materialized parameter observation in each step on "
        "each rank: [amax, whole-tensor L2 norm, presence]. Layer indexes a canonical FQN/expert "
        "catalog. Unobserved entries are [NaN, NaN, 0]. The tensor may already be gathered by "
        "training; its pre-update population differs from the owned post-update shard in "
        "monitor/weight_local. Existing monitor behavior is preserved. Ranks that did not observe "
        "the entry store no row, and the ranks of a node that observed it are stored once (rank -1) "
        "when they agree.",
        collapse_equal_ranks=True,
        presence_index=2,
    ),
    MetricSpec(
        metric_id=171,
        name="attn/softmax/lse_stats",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Softmax-attention log-partition of every query row of every head, per rank, layer "
        "and microbatch: [rows, lse_sum x H, lse_min x H, lse_max x H], width=1+3H, all heads of a "
        "field together. The log-partition is the natural-log LSE of softmax_scale*q.k over the keys "
        "the causal and document mask allows, returned by the flash-attention forward. Sums are "
        "float64 sums over the counted rows (mean = sum / rows, rows exact up to 2^24); a non-finite "
        "value makes the statistics of its head non-finite. All attention calls of the microbatch "
        "contribute.",
    ),
    MetricSpec(
        metric_id=172,
        name="attn/softmax/qkv_stats",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="value",
        description="Softmax-attention query, key and value magnitudes of every token and head, per "
        "rank, layer and microbatch: [rows, q_nope_norm_sum x H, q_rope_norm_sum x H, "
        "k_nope_norm_sum x H, k_rope_norm_sum x H, v_norm_sum x H, q_amax x H, k_amax x H, "
        "v_amax x H], width=1+8H. A norm sum is the float64 sum over tokens of the L2 norm of the "
        "head's slice (the position-free or rotary part of q and k, the whole of v); an amax is the "
        "largest absolute element over all tokens of the whole head slice (both parts of q and k). "
        "Rows are tokens, exact up to 2^24; a non-finite value makes the statistics of its head "
        "non-finite. All attention calls of the microbatch contribute.",
    ),
    MetricSpec(
        metric_id=173,
        name="attn/softmax/gate_stats",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="logit",
        description="Output-gate logits of softmax attention, per rank, layer and microbatch: [rows, "
        "gate_sum x H, gate_min x H, gate_max x H], width=1+3H, of each token's mean over the value "
        "width of the head's gate logit (the gate projection before the sigmoid). Layers without "
        "gated attention have no row. Rows are tokens, exact up to 2^24; sums are float64; a "
        "non-finite value makes the statistics of its head non-finite.",
    ),
    MetricSpec(
        metric_id=174,
        name="attn/softmax/row_stats",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="nats",
        description="Softmax-attention distribution of every query row of every head with at least 32 "
        "allowed keys, per rank, layer and microbatch: [rows, ent_sum, ent_sumsq, ent_min, ent_max, "
        "entn_sum, entn_min, entn_max, pmax_sum, pmax_min, pmax_max, pmin_sum, lpmin_sum, lpmin_min, "
        "lpmin_max, mlogit_sum, mlogit_max, coll_sum, sink_sum, sink_max, dist_sum, local_sum], each "
        "field x H, width=1+21H. With x = softmax_scale*q.k - LSE over the allowed keys and p = exp(x): "
        "ent = -sum p x (nats), entn = ent / log(keys), pmax = exp(max x), pmin = exp(min x), lpmin = "
        "min x, mlogit = the largest logit, coll = sum p^2, sink = p of the document's first key, "
        "dist = sum p (i - j), local = sum of p over the last 128 keys. Exact over every row; sums are "
        "float64 (mean = sum / rows).",
    ),
    MetricSpec(
        metric_id=175,
        name="attn/softmax/entropy_hist",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="rows",
        description="Per-layer histogram of the normalized softmax-attention entropy entn of every "
        "query row with at least 32 allowed keys, pooled over heads, every step, per rank and "
        "microbatch.",
        bins=ROUTER_ENTROPY_BINS,
    ),
    MetricSpec(
        metric_id=176,
        name="attn/softmax/head_entropy_hist",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="rows",
        description="attn/softmax/entropy_hist separately for each head, head-major (32 bins of head 0, "
        "then head 1, ...), width=32H; detailed-mode only.",
        bins=ROUTER_ENTROPY_BINS,
        detailed=True,
    ),
    *_quantile_metrics(
        first_metric_id=177,
        quantiles_name="attn/softmax/entropy_quantiles",
        counts_name="attn/softmax/entropy_quantile_counts",
        population_description="Normalized softmax-attention entropy of every query row with at least 32 allowed keys, "
        "pooled over heads.",
    ),
    MetricSpec(
        metric_id=179,
        name="attn/linear/gate_stats",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="value",
        description="Gated delta-rule gates of every token and value head, per rank, layer and "
        "microbatch: [rows, beta_sum x Hv, beta_min x Hv, beta_max x Hv, alpha_sum x Hv, "
        "alpha_min x Hv, alpha_max x Hv, g_sum x Hv], width=1+7Hv. beta is the write strength the "
        "kernel receives (the sigmoid, doubled when negative eigenvalues are allowed), g the "
        "log-decay it receives and alpha = exp(g) the decay of the state. Rows are tokens, exact "
        "up to 2^24; sums are float64; a non-finite value makes the statistics of its head "
        "non-finite.",
    ),
    MetricSpec(
        metric_id=180,
        name="attn/linear/norm_stats",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="value",
        description="Value and output magnitudes of gated delta-rule attention per token and value "
        "head, per rank, layer and microbatch: [rows, v_norm_sum x Hv, o_norm_sum x Hv, "
        "o_amax x Hv], width=1+3Hv. v is the value after the short convolution and o the kernel "
        "output before the output norm and gate. A norm sum is the float64 sum over tokens of the "
        "L2 norm of the head's slice; o_amax is the largest absolute element over all tokens. Rows "
        "are tokens, exact up to 2^24; a non-finite value makes the statistics of its head "
        "non-finite.",
    ),
    MetricSpec(
        metric_id=181,
        name="attn/linear/state_stats",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.PER_RANK,
        unit="value",
        description="Final recurrent state of gated delta-rule attention per document and value head, "
        "per rank, layer and microbatch: [documents, state_norm_sum x Hv, state_norm_max x Hv], "
        "width=1+2Hv, of the Frobenius norm of each document's final state; sums are float64.",
    ),
    MetricSpec(
        metric_id=182,
        name="attn/linear/decay_params",
        family=Family.ROUTER_VECTORS,
        value_type=ValueType.FLOAT32,
        node_merge=NodeMerge.IDENTICAL,
        unit="value",
        description="Decay parameters of each gated delta-rule layer's value heads, once per step "
        "(microbatch -1), identical on every rank: [A x Hv, softplus(dt_bias) x Hv], width=2Hv, "
        "with A = exp(A_log). The per-token log-decay is -A * softplus(a + dt_bias).",
        job_identical=True,
    ),
    MetricSpec(
        metric_id=187,
        name="router/stream_tokens_per_expert",
        family=Family.EXPERT_COUNTS,
        value_type=ValueType.INT32,
        node_merge=NodeMerge.PER_RANK,
        unit="tokens",
        description="Rank-local original top-k assignments per expert and data stream for one microbatch, "
        "laid out [expert][stream]. Counted at router selection before capacity filtering, including input-padding "
        "rows and regardless of label validity, before EP exchange or monitor reductions.",
        required=True,
    ),
)

_METRICS_BY_NAME = {metric.name: metric for metric in METRICS}
_METRICS_BY_ID = {metric.metric_id: metric for metric in METRICS}


def metric_by_name(name: str) -> MetricSpec:
    return _METRICS_BY_NAME[name]


def metric_by_id(metric_id: int) -> MetricSpec:
    return _METRICS_BY_ID[metric_id]
