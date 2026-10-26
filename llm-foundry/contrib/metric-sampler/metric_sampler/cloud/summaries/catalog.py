from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CHUNK_STEPS = 10

_SELECTED = "exp_filter = '__EXP__', run_filter = '__RUN__', rank_filter = -2, microbatch_filter = -2"
_RANGE = "step_from = __FROM__, step_to = __TO__"
_SELECTED_RANGE = f"{_SELECTED}, {_RANGE}"
_RUN_RANGE = f"exp_filter = '__EXP__', run_filter = '__RUN__', {_RANGE}"
_RUN_FROM = "exp_filter = '__EXP__', run_filter = '__RUN__', step_from = __FROM__"
_ALL_LAYERS_RANGE = f"{_SELECTED_RANGE}, layer_filter = -1"

_LOSS_FAMILIES = (
    "loss_quantiles", "stream_loss", "stream_share", "expert_loss", "loss_position", "sample_loss_quantiles",
)

_INSERT_TEMPLATE = (
    "INSERT INTO metric_sampler.summary_{name} SELECT now64(3), * FROM ({select}) "
    "SETTINGS max_threads = {max_threads}, priority = 10, max_execution_time = 1800, "
    "max_memory_usage = 30000000000, "
    "max_bytes_before_external_group_by = 10000000000, optimize_move_to_prewhere_if_final = 1, "
    "log_comment = '{log_comment}-{name}'"
)


def _view(name: str, arguments: str) -> str:
    return f"SELECT * FROM metric_sampler.{name}({arguments}) WHERE __WHERE__"


@dataclass(frozen=True)
class Summary:
    """One summary table and how its loop fills it.

    `source` is the view query with __EXP__, __RUN__, __FROM__ and __TO__ standing for the chunk and __WHERE__ for the
    step and run filter. `after` names a summary this one is derived from: the chunk source reads that summary's rows,
    so the loop computes only steps its progress has reached. `progress` lists the summaries whose progress the loop
    keeps, its own by default: a loop whose insert fills several summary tables, through the views of the table it
    inserts into, resumes from their lowest progress and moves them all. A chunk's memory grows with its steps and
    with the rows of each step, so `chunk_steps` is smaller for the summaries that hold more per step.
    """

    name: str
    source: str
    chunk_steps: int = DEFAULT_CHUNK_STEPS
    after: str | None = None
    progress: tuple[str, ...] = ()

    @property
    def progress_names(self) -> tuple[str, ...]:
        return self.progress or (self.name,)

    @property
    def bounded_to_chunk(self) -> bool:
        """Whether the source takes the chunk's last step, so that only rows of the chunk's steps change it."""
        return "__TO__" in self.source

    def chunk_select(self, exp: str, run: str, first: int, last: int) -> str:
        where = f"exp = '{exp}' AND run = '{run}' AND step BETWEEN {first} AND {last}"
        return (
            self.source.replace("__EXP__", exp)
            .replace("__RUN__", run)
            .replace("__FROM__", str(first))
            .replace("__TO__", str(last))
            .replace("__WHERE__", where)
        )

    def chunk_insert(self, exp: str, run: str, first: int, last: int, max_threads: int, log_comment: str) -> str:
        return _INSERT_TEMPLATE.format(
            name=self.name,
            select=self.chunk_select(exp, run, first, last),
            max_threads=max_threads,
            log_comment=log_comment,
        )


# Per-rank and per-microbatch summaries serve every run, so a rank or microbatch selection never falls back to the
# raw tables. The loops whose chunks hold more per step take fewer steps: the layer loads keep every rank's and
# microbatch's dispatcher counts of a chunk in memory, and the activation statistics per rank and microbatch are
# larger still. The activation statistics per rank, per microbatch and of all ranks sum the finished rows of the per
# rank and microbatch ones (`after`) instead of reading the raw rows again.
# The four levels of the massive tokens and of the exact quantiles (all, per rank, per microbatch, per rank and
# microbatch) come from one read of the histograms: one loop per family fills the four summaries and keeps their
# progress. A chunk holds the sums of all four levels at once, hence the chunks of 3 steps.
SUMMARIES = (
    Summary("layer_load", _view("layer_load_selected_range", _SELECTED_RANGE), chunk_steps=5),
    Summary("expert_totals", _view("expert_totals_selected_range", _SELECTED_RANGE)),
    Summary("activation", _view("activation_selected_range", _SELECTED_RANGE), after="activation_rank_mb"),
    *(Summary(family, _view(f"{family}_per_step_range", _RUN_RANGE)) for family in _LOSS_FAMILIES),
    Summary("grad_per_rank", _view("grad_per_rank_run", _RUN_FROM)),
    Summary("timing_per_rank", _view("timing_per_rank_run", _RUN_FROM)),
    Summary(
        "router_rank",
        "SELECT exp, run, attempt, node, step, layer, rank, microbatch, metric_id, values "
        "FROM metric_sampler.expert_counts FINAL WHERE layer <= 255 AND metric_id IN (1, 2) AND __WHERE__",
    ),
    Summary("capacity", _view("capacity_selected_range", _SELECTED_RANGE)),
    Summary(
        "activation_rank", _view("activation_by_rank_range", _SELECTED_RANGE), chunk_steps=5,
        after="activation_rank_mb",
    ),
    Summary("activation_mb", _view("activation_by_mb_range", _SELECTED_RANGE), after="activation_rank_mb"),
    Summary("activation_rank_mb", _view("activation_by_rank_mb_range", _SELECTED_RANGE), chunk_steps=3),
    Summary(
        "activation_massive_levels",
        _view("activation_massive_levels_range", _SELECTED_RANGE),
        chunk_steps=3,
        progress=(
            "activation_massive", "activation_massive_rank", "activation_massive_mb", "activation_massive_rank_mb",
        ),
    ),
    Summary(
        "activation_exp_quantiles_levels",
        _view("activation_exp_quantiles_levels_range", _SELECTED_RANGE),
        chunk_steps=3,
        progress=(
            "activation_exp_quantiles", "activation_exp_quantiles_rank", "activation_exp_quantiles_mb",
            "activation_exp_quantiles_rank_mb",
        ),
    ),
    Summary("capacity_rank", _view("capacity_by_rank_range", _SELECTED_RANGE)),
    Summary("capacity_mb", _view("capacity_by_mb_range", _SELECTED_RANGE)),
    Summary("capacity_rank_mb", _view("capacity_by_rank_mb_range", _SELECTED_RANGE)),
    Summary("router_balance", _view("router_balance_range", _SELECTED_RANGE)),
    Summary("router_balance_rank", _view("router_balance_by_rank_range", _SELECTED_RANGE)),
    Summary("router_balance_mb", _view("router_balance_by_mb_range", _SELECTED_RANGE)),
    Summary("router_balance_rank_mb", _view("router_balance_by_rank_mb_range", _SELECTED_RANGE)),
    Summary("weight_stats", _view("weight_stats_range", _SELECTED_RANGE)),
    Summary("module_activation", _view("module_activation_range", _SELECTED_RANGE)),
    Summary("expert_stream", _view("expert_stream_range", _SELECTED_RANGE), chunk_steps=100),
    Summary("layer_load_mb", _view("layer_load_by_mb_range", _SELECTED_RANGE), chunk_steps=5),
    Summary("layer_load_rank", _view("layer_load_rank_range", _RUN_RANGE)),
    Summary("expert_totals_mb", _view("expert_totals_by_mb_range", _SELECTED_RANGE)),
    Summary("activation_amax", _view("activation_amax_range", _SELECTED_RANGE)),
    Summary("activation_amax_mb", _view("activation_amax_mb_range", _SELECTED_RANGE)),
    Summary("loss_components", _view("loss_components_range", _SELECTED_RANGE)),
    Summary("loss_rank", _view("loss_rank_range", _SELECTED_RANGE)),
    Summary("loss_spread", _view("loss_spread_range", _SELECTED_RANGE)),
    Summary("loss_rank_mb", _view("loss_rank_mb_range", _RUN_RANGE)),
    Summary("loss_step", _view("loss_per_step_range", _SELECTED_RANGE)),
    Summary("step_time", _view("step_time_range", _SELECTED_RANGE)),
    Summary("attention_softmax", _view("attention_softmax_range", _ALL_LAYERS_RANGE), chunk_steps=5),
    Summary("attention_linear", _view("attention_linear_range", _ALL_LAYERS_RANGE)),
    Summary("attention_rank", _view("attention_rank_range", _ALL_LAYERS_RANGE), chunk_steps=5),
    Summary("attention_entropy_hist", _view("attention_entropy_hist_range", _ALL_LAYERS_RANGE)),
)
SUMMARIES_BY_NAME = {summary.name: summary for summary in SUMMARIES}
