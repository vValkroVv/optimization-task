from dataclasses import dataclass, field

from torchtitan.config.schema.protocols.types import StrEnum

AxisName = str | int
DimSelector = AxisName | list[AxisName] | tuple[AxisName, ...]
ReduceGroup = str | tuple[str, ...]


class ReduceGroupName(StrEnum):
    """Group names a ``ReduceStageConfig`` may use.

    ``MeshAxisName`` values (kept in sync by a test), the derived meshes and
    the ``_GROUP_ALIASES``. Configs carry plain strings, checked against this
    set at construction so a typo fails before training.
    """

    dp = "dp"
    dp_replicate = "dp_replicate"
    dp_shard = "dp_shard"
    fsdp = "fsdp"
    tp = "tp"
    cp = "cp"
    pp = "pp"
    ep = "ep"
    efsdp = "efsdp"
    # Derived meshes.
    loss = "loss"
    batch = "batch"
    world = "world"
    # Aliases ("global" is a Python keyword, hence the trailing underscore).
    sp = "sp"
    sequence = "sequence"
    data = "data"
    global_ = "global"


def _validate_reduce_group(group: ReduceGroup | None) -> None:
    names = group if isinstance(group, tuple) else (group,)
    for name in names:
        if name is None:
            continue
        try:
            ReduceGroupName(name)
        except ValueError:
            raise ValueError(
                f"Unknown reduce group {name!r}; valid groups: "
                f"{[member.value for member in ReduceGroupName]}."
            ) from None


class ReduceMode(StrEnum):
    online = "online"
    offline = "offline"


class ReduceOp(StrEnum):
    mean = "mean"
    sum = "sum"
    max = "max"
    min = "min"
    none = "none"


class DTensorMode(StrEnum):
    local = "local"
    full_tensor = "full_tensor"
    partial_stats = "partial_stats"


class MetricStepKind(StrEnum):
    stat = "stat"
    reduce = "reduce"


class ThresholdCmp(StrEnum):
    gt = "gt"
    lt = "lt"


@dataclass(frozen=True, kw_only=True, slots=True)
class AxisSpec:
    names: tuple[AxisName, ...] = ()

    def resolve(self, dim: DimSelector | None) -> int | tuple[int, ...] | None:
        """Map axis names to positional dims, mirroring the input's shape.

        ``None`` passes through, a single name or index resolves to an
        ``int``, and a sequence resolves to a ``tuple`` of ints -- matching
        what torch reduction ops accept for ``dim``.
        """
        if dim is None:
            return None
        if isinstance(dim, (str, int)):
            return self._index_of(dim)
        if not dim:
            raise ValueError(
                "Empty axis selection is ambiguous; use None for a full reduction."
            )
        return tuple(self._index_of(axis) for axis in dim)

    def _index_of(self, axis: AxisName) -> int:
        if isinstance(axis, int):
            return axis
        try:
            return self.names.index(axis)
        except ValueError:
            raise ValueError(
                f"Unknown axis '{axis}'. Available axes: {self.names}"
            ) from None


class ExpansionCache:
    """A slot for the expanded-plan cache."""

    __slots__ = ("_expanded",)


@dataclass(frozen=True, kw_only=True, slots=True)
class ReduceStageConfig:
    """One distributed reduction: ``op`` applied over the ranks of ``group``."""

    group: ReduceGroup | None = None
    op: ReduceOp = ReduceOp.mean

    def __post_init__(self) -> None:
        _validate_reduce_group(self.group)


@dataclass(frozen=True, kw_only=True, slots=True)
class ReduceConfig(ExpansionCache):
    """Distributed reduction plan for a computed metric value.

    ``stages`` are applied in order, so a config can express cases like
    "mean within each FSDP group, then max across TP ranks". ``group``/``op``
    remain as shorthand for simple one-stage reductions and are ignored when
    ``stages`` is set. Reductions never span the PP axis: per-stage metrics
    are merged by ``gather_metrics_over_pp`` instead.
    """

    mode: ReduceMode = ReduceMode.online
    stages: list[ReduceStageConfig] = field(default_factory=list)
    # Shorthand for the common single-stage case.
    group: ReduceGroup | None = None
    op: ReduceOp = ReduceOp.mean

    def __post_init__(self) -> None:
        _validate_reduce_group(self.group)

    def effective_stages(self) -> tuple[ReduceStageConfig, ...]:
        """The stages to run, expanding the ``group``/``op`` shorthand.

        Cached as a tuple: this runs once per captured tensor per metric. The
        cache misses in-place edits of ``stages``; ``dataclasses.replace``
        drops it.
        """
        cached = getattr(self, "_expanded", None)
        if cached is None:
            cached = self._expand()
            object.__setattr__(self, "_expanded", cached)
        return cached

    def _expand(self) -> tuple[ReduceStageConfig, ...]:
        if self.stages:
            return tuple(self.stages)
        if self.group is None or self.op == ReduceOp.none:
            return ()
        return (ReduceStageConfig(group=self.group, op=self.op),)


class StatOpName(StrEnum):
    ABS = "abs"
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"
    ABS_MAX = "abs_max"
    RMS = "rms"
    NORM = "norm"
    COUNT = "count"
    NONZERO_COUNT = "nonzero_count"
    ZERO_RATIO = "zero_ratio"
    FINITE_RATIO = "finite_ratio"
    NAN_COUNT = "nan_count"
    INF_COUNT = "inf_count"
    QUANTILE = "quantile"
    KURTOSIS = "kurtosis"
    ACTABS = "actabs"
    ACTREL = "actrel"
    ENTROPY = "entropy"
    RAW_ENTROPY_NORM = "raw_entropy_norm"
    MAX_MIN = "max_min"
    MAX_VIO = "max_vio"
    MAX_MIN_RATIO = "max_min_ratio"
    MIN_MEDIAN_RATIO = "min_median_ratio"
    POSITIVE_RATIO = "positive_ratio"
    CV = "cv"
    GINI = "gini"
    BINCOUNT = "bincount"


@dataclass(frozen=True, kw_only=True, slots=True)
class StatOpConfig:
    """One statistic in a metric pipeline."""

    op: str
    dim: DimSelector | None = None
    q: float | None = None
    p: float | None = None
    minlength: int | None = None
    threshold: float | None = None
    cmp: ThresholdCmp = ThresholdCmp.gt

    def __post_init__(self) -> None:
        StatOpName(self.op)


@dataclass(frozen=True, kw_only=True, slots=True)
class MetricStepConfig:
    """A tagged step in a metric plan: a local stat or a distributed reduce."""

    kind: MetricStepKind
    stat: StatOpConfig | None = None
    reduce: ReduceStageConfig | None = None

    def __post_init__(self) -> None:
        if self.kind == MetricStepKind.stat and self.stat is None:
            raise ValueError("MetricStepConfig(kind='stat') requires 'stat'")
        if self.kind == MetricStepKind.reduce and self.reduce is None:
            raise ValueError("MetricStepConfig(kind='reduce') requires 'reduce'")


@dataclass(frozen=True, kw_only=True, slots=True)
class MetricConfig(ExpansionCache):
    """Full plan for one metric derived from a sample.

    ``steps`` interleaves stats and reductions explicitly; when it is empty,
    the ``pipeline`` + ``reduce`` shorthand expands to "all stats, then all
    reduction stages". As with ``ReduceConfig``, the explicit form wins and
    the shorthand fields are ignored when ``steps`` is set.
    """

    name: str
    steps: list[MetricStepConfig] = field(default_factory=list)
    pipeline: list[StatOpConfig] = field(default_factory=list)
    reduce: ReduceConfig = field(default_factory=ReduceConfig)
    dtensor_mode: DTensorMode = DTensorMode.partial_stats

    def effective_steps(self) -> tuple[MetricStepConfig, ...]:
        """The steps to run, expanding the ``pipeline``/``reduce`` shorthand.

        Cached on the same terms as :meth:`ReduceConfig.effective_stages`.
        """
        cached = getattr(self, "_expanded", None)
        if cached is None:
            cached = self._expand()
            object.__setattr__(self, "_expanded", cached)
        return cached

    def _expand(self) -> tuple[MetricStepConfig, ...]:
        if self.steps:
            return tuple(self.steps)
        return (
            *(
                MetricStepConfig(kind=MetricStepKind.stat, stat=stat)
                for stat in self.pipeline
            ),
            *(
                MetricStepConfig(kind=MetricStepKind.reduce, reduce=stage)
                for stage in self.reduce.effective_stages()
            ),
        )
