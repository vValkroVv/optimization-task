from dataclasses import dataclass

import torch
import torch.distributed as dist

from typing import TypeAlias

from torchtitan.config.schema.callbacks.sources.dispatcher_source import (
    DispatcherSourceConfig,
)
from torchtitan.config.schema.callbacks.sources.dispatcher_source import MaxVioConfig
from torchtitan.config.schema.callbacks.sources.dispatcher_source import (
    MaxVioLevel,
    MaxVioWindow,
)
from torchtitan.callbacks.base import ObserverState
from torchtitan.callbacks.metrics.sample import MetricRecord, Sample
from torchtitan.config.schema.callbacks.metrics.sample import (
    AxisSpec,
    ReduceConfig,
    StatOpConfig,
)
from torchtitan.callbacks.metrics.stats import (
    _local_tensor,
    apply_stat_op,
)
from torchtitan.callbacks.sources.base import (
    MonitorSource,
)
from torchtitan.callbacks.sources.moe_units import (
    MoEMonitorUnit,
    iter_moe_monitor_units,
)
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed.pipeline_context import is_pipeline_metadata_inference
from torchtitan.distributed.pipeline_parallel import in_metadata_probe
from torchtitan.protocols.moe_monitoring import DispatchObservation

__all__ = [
    "DispatcherSource",
    "MaxVioConfig",
    "MaxVioLevel",
    "MaxVioWindow",
]


def _gather_concat(
    tensor: torch.Tensor | None,
    parallel_dims: object,
    mesh_name: str,
    dim: int = 0,
) -> torch.Tensor | None:
    """Concatenate per-rank tensors over a mesh (per-local-expert loads over ep)."""
    if tensor is None:
        return None
    mesh = parallel_dims.get_optional_mesh(mesh_name)
    if mesh is None or mesh.size() <= 1:
        return tensor
    parts = [torch.empty_like(tensor) for _ in range(mesh.size())]
    dist.all_gather(parts, tensor.contiguous(), group=mesh.get_group())
    return torch.cat(parts, dim=dim)


@dataclass(slots=True)
class _UnitAccum:
    """One MoE unit's counters for the batch in progress."""

    # window key -> running histogram sum of this batch's forwards
    sums: dict[str, torch.Tensor]
    # window key -> (forwards_per_batch, E) block, one row per forward
    rows: dict[str, torch.Tensor]
    # Next row to write; doubles as the forward count the log step verifies.
    cursor: torch.Tensor
    # This rank's tokens per expert, summed over the batch's forwards.
    local: torch.Tensor
    # Rows dispatched to this rank, summed over the batch's forwards.
    routed: torch.Tensor
    drop: torch.Tensor | None = None

    def zero_(self) -> None:
        self.drop = None
        torch._foreach_zero_(
            [
                *self.sums.values(),
                *self.rows.values(),
                self.cursor,
                self.local,
                self.routed,
            ]
        )


@dataclass(slots=True)
class _UnitWindow:
    """One MoE unit's logging window, preallocated to ``log_interval`` batches."""

    # window key -> (log_interval, E): one summed histogram per batch
    sums: dict[str, torch.Tensor]
    # window key -> (log_interval, forwards_per_batch, E)
    rows: dict[str, torch.Tensor]
    # Batches folded since the last consumer
    filled: int = 0

    def capacity(self) -> int:
        return next((t.shape[0] for t in (*self.sums.values(), *self.rows.values())), 0)

    def batches(self, buf: torch.Tensor) -> torch.Tensor:
        return buf[: min(self.filled, buf.shape[0])]

    def fold(self, accum: _UnitAccum) -> None:
        capacity = self.capacity()
        if capacity == 0:
            return
        slot = self.filled % capacity
        for key, buf in self.sums.items():
            buf[slot].copy_(accum.sums[key])
        for key, buf in self.rows.items():
            buf[slot].copy_(accum.rows[key])
        self.filled += 1


class DispatcherSource(MonitorSource):
    """Token-dispatcher load statistics of every MoE layer.

    A dispatcher is not an ``nn.Module``, so ``dispatch()`` cannot carry a
    forward hook; ``fit_start`` assigns its ``dispatch_monitor`` slot instead,
    and ``close`` clears it. Nothing accumulates model-side. Counts are
    collected on every batch — not only logged ones — because max_vio averages
    the whole logging window. Per-expert loads are gathered over ep, and
    metrics of every pipeline stage reach the logging stage through the
    emitter, so the logged values describe the full expert set. Histograms,
    batch accumulators, window buffers, and count collectives use int64;
    max_vio converts to float64 only after the count reductions.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = DispatcherSourceConfig

    config: Config

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.units: list[MoEMonitorUnit] = []
        self._accum: dict[str, _UnitAccum] = {}
        self._window: dict[str, _UnitWindow] = {}
        # Batches one logging window spans; MonitoringCallback assigns it
        # before fit_start. Sizes the window buffers.
        self.log_interval = 1
        # The forwards derivation is checked once, on the process's first
        # batch: the check is a host sync.
        self._forwards_checked = False
        # Forwards one optimizer batch runs on this rank; sizes the row
        # blocks. Resolved in fit_start.
        self._forwards_per_batch = 1
        # Window keys that must be kept, resolved against the masks: a series
        # whose mask is unset reads "global", so that is the key it needs.
        self._needs_sums: set[str] = set()
        self._needs_rows: set[str] = set()
        # Per-token masks of the CURRENT forward, (T,) bool, True = keep.
        self._padding_mask_T: torch.Tensor | None = None
        self._prefix_mask_T: torch.Tensor | None = None
        # Masks built at after_dataloader (one event per microbatch, all before
        # the first forward), consumed by microbatch_start in the same order.
        self._mask_queue: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._masks_enabled = False
        # Step the cached max_vio below was computed for; -1 means "no cache".
        self._max_vio_step = -1
        self._max_vio_cache: dict[str, dict[str, float]] = {}

    def fit_start(self, state: ObserverState) -> None:
        units = iter_moe_monitor_units(state.model_parts)
        kept = self._filter_fqns([unit.name for unit in units], "MoE units")
        self.units = [unit for unit in units if unit.name in kept]

        masked = [s for s in self.config.max_vio if s.window != MaxVioWindow.global_]
        self._masks_enabled = bool(masked)
        self._padding_mask_T = None
        self._prefix_mask_T = None
        self._mask_queue.clear()

        pp_splits = 1
        if state.parallel_dims.pp_enabled:
            pp_splits = (
                state.config.training.local_batch_size
                // state.config.parallelism.pipeline_parallel_microbatch_size
            )
        self._forwards_per_batch = state.gradient_accumulation_steps * pp_splits
        self._pp_splits = pp_splits
        self._forwards_checked = False

        self._needs_sums = {MaxVioWindow.global_}
        self._needs_rows = set()
        for series in self.config.max_vio:
            key = self._resolved_window(series.window)
            if series.level == MaxVioLevel.batch:
                self._needs_sums.add(key)
            else:
                self._needs_rows.add(key)

        ep_mesh = state.parallel_dims.get_optional_mesh("ep")
        ep_size = ep_mesh.size() if ep_mesh is not None else 1
        self._accum.clear()
        self._window.clear()
        for unit in self.units:
            if unit.dispatcher is not None:
                accum = self._allocate(unit, ep_size=ep_size, device=state.device)
                self._accum[unit.name] = accum
                self._window[unit.name] = _UnitWindow(
                    sums={
                        key: t.new_zeros(self.log_interval, *t.shape)
                        for key, t in accum.sums.items()
                    },
                    rows={
                        key: t.new_zeros(self.log_interval, *t.shape)
                        for key, t in accum.rows.items()
                    },
                )
                unit.dispatcher.dispatch_monitor = self._observe(accum)

    def close(self) -> None:
        super().close()
        for unit in self.units:
            if unit.dispatcher is not None:
                unit.dispatcher.dispatch_monitor = None
        self.units = []
        self._accum.clear()
        self._window.clear()

    def _masked_histogram(
        self, observation: DispatchObservation, mask_T: torch.Tensor | None
    ) -> torch.Tensor | None:
        """The global histogram with the masked-out tokens removed."""
        if mask_T is None:
            return None
        ids_TK = observation.topk_expert_ids_TK
        if mask_T.numel() != ids_TK.shape[0]:
            raise ValueError(
                f"DispatcherSource: token mask has {mask_T.numel()} entries but "
                f"the routing map has {ids_TK.shape[0]} tokens; the mask must be "
                "one entry per dispatched token."
            )
        num_experts = int(observation.local_tokens_per_expert.numel())
        flat = ids_TK.detach().to(torch.long).reshape(-1)
        flat = flat.masked_fill((flat < 0) | (flat >= num_experts), num_experts)
        weights = mask_T.reshape(-1, 1).to(torch.int64).expand(ids_TK.shape).reshape(-1)
        out = torch.zeros(num_experts + 1, dtype=torch.int64, device=flat.device)
        return out.scatter_add_(0, flat, weights)[:num_experts]

    def _resolved_window(self, window: MaxVioWindow) -> MaxVioWindow:
        """The window key a series actually reads."""
        if window == MaxVioWindow.global_ or not self._masks_enabled:
            return MaxVioWindow.global_
        return window

    def _observe(self, accum: _UnitAccum):
        """Fold one ``dispatch()`` into the batch in progress."""
        sums = accum.sums
        rows = accum.rows
        cursor, local, routed = accum.cursor, accum.local, accum.routed

        def observe(observation: DispatchObservation) -> None:
            if self._is_recompute() or in_metadata_probe():
                return
            if not torch.compiler.is_compiling() and is_pipeline_metadata_inference():
                return
            with torch.no_grad():
                hists = {
                    # Keep routed counts exact when accumulating microbatches.
                    MaxVioWindow.global_: _local_tensor(
                        observation.global_tokens_per_local_expert,
                        full_tensor=True,
                        dtype=torch.int64,
                    )
                }
                pad = self._padding_mask_T
                if (
                    pad is not None
                    and pad.shape[-1] == observation.topk_expert_ids_TK.shape[0]
                ):
                    # This forward's PP chunk; a single row when pp_splits=1.
                    row = (cursor % pad.shape[0]).reshape(1)
                    pad_T = pad.index_select(0, row).reshape(-1)
                    prefix_T = self._prefix_mask_T.index_select(0, row).reshape(-1)
                    hists[MaxVioWindow.no_pad] = self._masked_histogram(
                        observation, pad_T
                    )
                    hists[MaxVioWindow.no_pad_no_prefix] = self._masked_histogram(
                        observation, pad_T & prefix_T
                    )
                elif self._masks_enabled:
                    keep_all = torch.ones_like(
                        observation.topk_expert_ids_TK[:, 0], dtype=torch.bool
                    )
                    full = self._masked_histogram(observation, keep_all)
                    hists[MaxVioWindow.no_pad] = full
                    hists[MaxVioWindow.no_pad_no_prefix] = full

                for key, total in sums.items():
                    total.add_(hists[key])
                if rows:
                    # One row per forward; clamped so the traced index stays in
                    # range, the first batch's check catches a wrong count.
                    index = cursor.reshape(1).clamp(max=self._forwards_per_batch - 1)
                    for key, block in rows.items():
                        block.index_copy_(0, index, hists[key].unsqueeze(0))
                cursor.add_(1)
                local.add_(
                    _local_tensor(
                        observation.local_tokens_per_expert,
                        full_tensor=True,
                        dtype=torch.int64,
                    )
                )
                routed.add_(hists[MaxVioWindow.global_].sum())
                if not torch.compiler.is_compiling() and self.active:
                    scores = observation.requested_topk_scores
                    keep = observation.capacity_keep_mask
                    if scores is not None and keep is not None:
                        requested = scores != 0
                        retained = requested & keep
                        tokens = requested.any(-1)
                        drop = torch.stack(
                            (
                                requested.sum(),
                                retained.sum(),
                                tokens.sum(),
                                (tokens & ~retained.any(-1)).sum(),
                                scores.sum(dtype=torch.float64),
                                (scores * ~keep).sum(dtype=torch.float64),
                            )
                        ).double()
                        if accum.drop is None:
                            accum.drop = drop
                        else:
                            accum.drop.add_(drop)

        return observe

    def _allocate(self, unit: MoEMonitorUnit, *, ep_size: int, device) -> _UnitAccum:
        """One unit's accumulators, sized from the routing contract."""
        num_experts = int(unit.router.num_experts)
        sizes = dict.fromkeys(self._needs_sums | self._needs_rows, num_experts)
        sizes[MaxVioWindow.global_] = num_experts // ep_size

        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, dtype=torch.int64, device=device)

        return _UnitAccum(
            sums={key: zeros(sizes[key]) for key in self._needs_sums},
            rows={
                key: zeros(self._forwards_per_batch, sizes[key])
                for key in self._needs_rows
            },
            cursor=torch.zeros((), dtype=torch.long, device=device),
            local=zeros(num_experts),
            routed=torch.zeros((), dtype=torch.long, device=device),
        )

    def after_dataloader(self, state: ObserverState) -> None:
        """Build this microbatch's masks; one event per microbatch, pre-forward."""
        if not self._masks_enabled or state.labels is None:
            return
        labels = state.labels
        attention = (state.input_dict or {}).get("attention_mask")
        padding = (
            attention.reshape(-1).to(torch.bool)
            if attention is not None
            else torch.ones(labels.numel(), dtype=torch.bool)
        )
        self._mask_queue.append((padding, labels.reshape(-1) != IGNORE_INDEX))

    def microbatch_start(self, state: ObserverState) -> None:
        """Point the observer at this forward's masks before it runs."""
        if not self._mask_queue:
            return
        padding, prefix = self._mask_queue.pop(0)
        padding = padding.reshape(self._pp_splits, -1)
        prefix = prefix.reshape(self._pp_splits, -1)
        buffer = self._padding_mask_T
        if buffer is not None and padding.shape == buffer.shape:
            buffer.copy_(padding)
            self._prefix_mask_T.copy_(prefix)
        else:
            device = state.device if state.device is not None else padding.device
            self._padding_mask_T = padding.to(device)
            self._prefix_mask_T = prefix.to(device)

    def batch_start(self, state: ObserverState) -> None:
        super().batch_start(state)
        self._mask_queue.clear()
        for accum in self._accum.values():
            accum.zero_()

    def _max_vio(
        self,
        microbatch_hists: list[torch.Tensor],
        parallel_dims: object,
        *,
        concat_over_ep: bool,
        level: MaxVioLevel,
        full_expert_space: bool = False,
    ) -> float | None:
        """Window-averaged (max-mean)/mean of a per-microbatch histogram list.

        ``level`` selects the aggregation:
          - ``"microbatch"``: violation of each microbatch, averaged;
          - ``"batch"``: one entry per optimizer batch (the window stores batch
            sums), the histogram summed over the data ranks, then violation,
            then averaged;
          - ``"ep_group"``: as microbatch, with the per-expert axis folded
            into ep_size device groups first (device imbalance).
        Only ``"batch"`` pools the histogram across data ranks; the other two
        take the plain mean of the per-rank scalars.
        """
        if not microbatch_hists:
            return None
        # Entries are (E,) histograms or (forwards, E) blocks — one block per
        # batch is how the window stores per-forward rows.
        stacked = torch.cat(
            [hist.reshape(-1, hist.shape[-1]) for hist in microbatch_hists]
        )  # [n_microbatches, E_local]
        ep_size = 1
        if concat_over_ep:
            ep_mesh = parallel_dims.get_optional_mesh("ep")
            if ep_mesh is not None and ep_mesh.size() > 1:
                ep_size = ep_mesh.size()
                if full_expert_space:
                    stacked = stacked.contiguous()
                    dist.all_reduce(stacked, group=ep_mesh.get_group())
                else:
                    stacked = _gather_concat(stacked, parallel_dims, "ep", dim=-1)

        if level == MaxVioLevel.ep_group:
            if ep_size <= 1:
                return None
            rows, num_experts = stacked.shape
            stacked = stacked.reshape(rows, ep_size, num_experts // ep_size).sum(-1)

        data_mesh = parallel_dims.get_optional_mesh("loss")
        pooled = level == MaxVioLevel.batch
        if pooled and data_mesh is not None and data_mesh.size() > 1:
            dist.all_reduce(stacked, group=data_mesh.get_group())

        vio = apply_stat_op(
            stacked.double(), AxisSpec(), StatOpConfig(op="max_vio", dim=-1)
        )
        avg = vio.mean()
        if not pooled and data_mesh is not None and data_mesh.size() > 1:
            dist.all_reduce(avg, group=data_mesh.get_group())
            avg = avg / data_mesh.size()
        return float(avg.item())

    def after_optimizer_step(self, state: ObserverState) -> None:
        del state
        if not self._forwards_checked:
            self._forwards_checked = True
            for name, (forwards, _) in self._read_counters().items():
                # A unit switched off by the gate dispatches nothing.
                if forwards and forwards != self._forwards_per_batch:
                    raise ValueError(
                        f"DispatcherSource: unit '{name}' dispatched {forwards} "
                        "times this batch, but fit_start derived "
                        f"{self._forwards_per_batch} forwards per batch (gradient "
                        "accumulation x pipeline microbatches); fix the "
                        "derivation, then rerun."
                    )

        for unit_name, accum in self._accum.items():
            self._window[unit_name].fold(accum)

    def _read_counters(self) -> dict[str, tuple[int, int]]:
        """``{unit: (forwards, routed)}`` — one host sync for every unit."""
        names = list(self._accum)
        if not names:
            return {}
        stacked = torch.stack(
            [torch.stack((self._accum[n].cursor, self._accum[n].routed)) for n in names]
        )
        return {n: (int(f), int(r)) for n, (f, r) in zip(names, stacked.cpu().tolist())}

    def max_vio_by_unit(self, state: ObserverState) -> dict[str, dict[str, float]]:
        """``{unit name: {leaf: value}}`` for this step, computed once."""
        if self._max_vio_step == state.step:
            return self._max_vio_cache

        parallel_dims = state.parallel_dims
        result: dict[str, dict[str, float]] = {}
        for unit in self.units:
            if unit.dispatcher is None:
                continue
            window = self._window.get(unit.name)
            filled = window.filled if window is not None else 0
            values: dict[str, float] = {}
            for variant in self.config.max_vio:
                key = self._resolved_window(variant.window)
                source = (
                    window.sums if variant.level == MaxVioLevel.batch else window.rows
                )
                buf = source.get(key) if window is not None else None
                value = self._max_vio(
                    [window.batches(buf)] if buf is not None and filled else [],
                    parallel_dims,
                    concat_over_ep=True,
                    level=variant.level,
                    full_expert_space=key != MaxVioWindow.global_,
                )
                if value is not None:
                    values[variant.name] = value
            if window is not None:
                window.filled = 0
            result[unit.name] = values

        self._max_vio_step = state.step
        self._max_vio_cache = result
        return result

    def batch_end(self, state: ObserverState) -> None:
        super().batch_end(state)
        if not self.active:
            return

        parallel_dims = state.parallel_dims
        max_vio = self.max_vio_by_unit(state)
        metrics: dict[str, float] = {}
        model_level: dict[str, float] = {}
        model_level_layers: dict[str, int] = {}
        # EP peers share the model's unit order within a pipeline stage, even
        # when their observation dictionaries were populated in a different order.
        drop_names = [
            unit.name
            for unit in self.units
            if unit.name in self._accum and self._accum[unit.name].drop is not None
        ]
        drop_by_unit: dict[str, list[float]] = {}
        if drop_names:
            drop_totals = torch.stack([self._accum[name].drop for name in drop_names])
            ep_mesh = parallel_dims.get_optional_mesh("ep")
            if ep_mesh is not None and ep_mesh.size() > 1:
                dist.all_reduce(drop_totals, group=ep_mesh.get_group())
            drop_by_unit = dict(zip(drop_names, drop_totals.tolist(), strict=True))

        counters = self._read_counters()
        for unit in self.units:
            accum = self._accum.get(unit.name)
            forwards, routed = counters.get(unit.name, (0, 0))
            if accum is None or forwards == 0:
                # Rank-symmetric: every rank of the stage skips in lockstep.
                continue
            local_sum = accum.local
            glob_sum = accum.sums[MaxVioWindow.global_]
            prefix = f"{self.config.namespace}/{unit.name}"
            drop = drop_by_unit.get(unit.name)
            if drop is not None:
                requested, retained, tokens, lost, mass, lost_mass = drop
                for leaf, value in {
                    "requested_assignments": requested,
                    "retained_assignments": retained,
                    "dropped_assignments": requested - retained,
                    "assignment_drop_fraction": (requested - retained)
                    / max(requested, 1),
                    "fully_dropped_token_fraction": lost / max(tokens, 1),
                    "dropped_score_mass_fraction": lost_mass / mass if mass else 0.0,
                }.items():
                    metrics[f"{prefix}/token_drop/{leaf}"] = value

            for leaf, vector, vector_metrics in (
                (
                    "local_tokens_per_expert",
                    local_sum,
                    self.config.local_tokens_metrics,
                ),
                (
                    "global_tokens_per_local_expert",
                    _gather_concat(glob_sum, parallel_dims, "ep"),
                    self.config.global_tokens_metrics,
                ),
            ):
                if vector is None or vector.numel() == 0 or not vector_metrics:
                    continue
                self._record(
                    Sample(
                        name=f"{prefix}/{leaf}",
                        value=vector.double(),
                        axes=AxisSpec(names=("expert",)),
                        metrics=vector_metrics,
                    )
                )

            for leaf, value in max_vio.get(unit.name, {}).items():
                metrics[f"{prefix}/{leaf}"] = value
                model_level[leaf] = model_level.get(leaf, 0.0) + value
                model_level_layers[leaf] = model_level_layers.get(leaf, 0) + 1

            # Counts, not routed_input.shape[0]: fp8/HybridEP pad the rows.
            metrics[f"{prefix}/routed_tokens"] = float(routed)

        pp_mesh = parallel_dims.get_optional_mesh("pp")
        leaves = tuple(variant.name for variant in self.config.max_vio)
        if pp_mesh is not None and pp_mesh.size() > 1:
            pooled = torch.tensor(
                [model_level.get(leaf, 0.0) for leaf in leaves]
                + [float(model_level_layers.get(leaf, 0)) for leaf in leaves],
                dtype=torch.float64,
                device=state.device,
            )
            dist.all_reduce(pooled, group=pp_mesh.get_group())
            totals_list = pooled[: len(leaves)].tolist()
            counts = pooled[len(leaves) :].tolist()
            model_level = {
                leaf: totals_list[i] for i, leaf in enumerate(leaves) if counts[i] > 0
            }
            model_level_layers = {
                leaf: int(counts[i]) for i, leaf in enumerate(leaves) if counts[i] > 0
            }
        for leaf, total in model_level.items():
            name = f"{self.config.namespace}/{self.config.model_level_leaf}/{leaf}"
            metrics[name] = total / model_level_layers[leaf]

        for name, value in metrics.items():
            self._records.append(
                MetricRecord(
                    name=name,
                    value=float(value),
                    step=self._step,
                    reduce=ReduceConfig(),
                )
            )
