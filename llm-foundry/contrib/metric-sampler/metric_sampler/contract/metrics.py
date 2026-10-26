from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

SCHEMA_VERSION = 1


class Family(Enum):
    """The Parquet file and table a metric's rows are stored in, named by the value."""

    SCALARS = "scalars"
    EXPERT_COUNTS = "expert_counts"
    ROUTER_VECTORS = "router_vectors"


class NodeMerge(Enum):
    """How the node aggregator combines the same key across the ranks of one node."""

    SUM = "sum"
    PER_RANK = "per_rank"
    IDENTICAL = "identical"


class ValueType(Enum):
    """Both variants occupy one 4-byte word in the capture buffer and ring payload."""

    INT32 = "int32"
    FLOAT32 = "float32"


@dataclass(frozen=True)
class HistogramBins:
    """A linear bin grid: bin i covers [start + i*width, start + (i+1)*width), and the last bin is open-ended
    above that."""

    start: float
    width: float
    count: int


@dataclass(frozen=True)
class MetricSpec:
    metric_id: int
    name: str
    family: Family
    value_type: ValueType
    node_merge: NodeMerge
    unit: str
    description: str
    bins: HistogramBins | None = None
    summarize_experts: bool = False
    detailed: bool = False
    # Same value on every node of the job; only the primary node stores the row.
    job_identical: bool = False
    required: bool = False
    # A rank's row equal word for word to its row of this metric (same layer and microbatch) is
    # stored without values, naming that metric in alias_metric_id.
    alias_of: str | None = None
    # PER_RANK only: when every expected rank of a node reports the same words, the node stores one
    # row (rank -1, identical, contributors = those ranks) instead of a row per rank.
    collapse_equal_ranks: bool = False
    # PER_RANK only: index of a 0/1 presence word; ranks reporting 0 store no row, and the node's
    # expected contributors count only the ranks that reported 1.
    presence_index: int | None = None


# The layer and microbatch of a value recorded outside the decoder layers or once per step.
NO_LAYER = -1
NO_MICROBATCH = -1
# A Float64 value is stored as its low and high Int32 words.
WORDS_PER_FLOAT64 = 2
# A count or token total stored in an Int32 word must not exceed this.
INT32_MAX = 2**31 - 1
# The layer column is Int16: a catalog whose indices are stored as layers (parameters, activations, learning-rate
# groups) must have fewer entries than this.
LAYER_IDENTITY_LIMIT = 2**15

TOKEN_LOSS_BINS = HistogramBins(start=0.0, width=0.125, count=256)
EXPERT_LOSS_BINS = HistogramBins(start=0.0, width=0.25, count=64)
ROUTER_ENTROPY_BINS = HistogramBins(start=0.0, width=1 / 32, count=32)
SAMPLE_LOSS_BINS = HistogramBins(start=0.0, width=0.125, count=128)
# router/entropy_hist covers every token: the routing kernel takes entropy rows at this stride.
ENTROPY_TOKEN_STRIDE = 1

# Number of position-in-sequence buckets for loss/position_sum and loss/position_count.
LOSS_POSITION_BUCKETS = 64
# loss/worst_sample_info keeps WORST_SAMPLE_COUNT samples of WORST_SAMPLE_FIELDS words each:
# [stream_id, shard_id, index_in_shard, loss_position, finite_tokens, nonfinite_tokens].
WORST_SAMPLE_COUNT = 4
WORST_SAMPLE_FIELDS = 6

# Grid over log2|x|: bin 0 also holds exact zeros and anything below 2^-40 (log2 of 0 is -inf, which floors and
# clamps into bin 0); the last bin is open above 2^23.
ACTIVATION_LOG2_BINS = HistogramBins(start=-40.0, width=1.0, count=64)
# act/<point>/counts: [nonfinite_count, finite_count], then the exponent histogram and the token RMS histogram.
ACTIVATION_COUNTS_WIDTH = 2 + 2 * ACTIVATION_LOG2_BINS.count
# Activation points observed in every decoder layer that has them.
LAYER_ACTIVATION_POINTS = (
    "residual",
    "attn_out",
    "mlp_out",
    "shared_expert_out",
    "attn_residual_contribution",
    "mlp_residual_contribution",
    "swiglu_in",
    "down_in",
)
# Activation points outside the decoder layers (layer -1).
STANDALONE_ACTIVATION_POINTS = ("final_norm", "embed")

QUANTILE_LEVELS = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
# Population counts paired with the quantiles: [finite, NaN, positive infinity, negative infinity, masked].
QUANTILE_COUNTS_WIDTH = 5
