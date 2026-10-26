from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any, TypeAlias

import torch
import torch.distributed as dist
from torch import nn

from torchtitan.config.schema.callbacks.builtin.dynamic_block_lr_rescaler import (
    DynamicBlockLRRescalerConfig,
)
from torchtitan.callbacks.base import (
    MutableState,
    Mutation,
    ObserverState,
    PostMutator,
)
from torchtitan.callbacks.builtin.monitoring import MonitoringCallback
from torchtitan.callbacks.sources.dispatcher_source import DispatcherSource
from torchtitan.config.schema.components.optimizer import ParamGroupConfig
from torchtitan.protocols.blocks import walk_segments
from torchtitan.config import Configurable
from torchtitan.callbacks.sources.moe_units import iter_moe_monitor_units

logger = logging.getLogger(__name__)


# Matches the decoder layer index in a parameter or module FQN,
# e.g. "layers.29.moe.experts.w13_E2FD".
def _block_indices(model_parts: Iterable[nn.Module]) -> dict[int, int]:
    """``id(submodule or parameter) -> index`` of the decoder block owning it."""
    owners: dict[int, int] = {}
    for part in model_parts:
        named = getattr(part, "named_segments", None)
        segments = named() if callable(named) else walk_segments(part)
        for segment in segments:
            if not segment.is_block or segment.group != "layers":
                continue
            index = int(segment.key)
            for module in segment.module.modules():
                owners[id(module)] = index
            for param in segment.module.parameters():
                owners[id(param)] = index
    return owners


class DynamicBlockLRRescaler(Configurable, PostMutator):
    # Writes scheduler.base_lrs and group["lr"] of the blocks it controls.
    mutates = frozenset({Mutation.lr})

    """Reduce the learning rate of individual decoder blocks on the fly.

    Consumes, but does not produce, monitoring: the per-layer max-violation
    signal is computed by :class:`DispatcherSource`, which this callback reads
    through ``max_vio_by_unit``. That method caches its result per step, so the
    ep all-gather and loss all-reduce behind it run once whether this callback
    or the monitoring emitter asks first — and this callback logs only its own
    ``dynamic_block_lr/*`` tags. The samples accumulate in a per-block rolling
    window, and a heuristic periodically decides whether the block's learning rate should
    be reduced: it fits a robust Theil-Sen slope to the window and fires
    when the relative trend exceeds ``slope_threshold``. When it fires for
    block ``i``, every parameter of decoder layer ``i`` has its effective LR
    multiplied by ``reduce_coef``.

    The dynamic per-block scale starts at 1.0 and never drops below
    ``lr_scaling_floor``. No rescaling happens before ``warmup_steps`` (the
    noisy early phase of training). A block fires at most once per
    ``cooldown`` samples. The rolling window, the scales and the fire markers
    are checkpointed: a resumed run keeps accumulating toward a full window
    instead of restarting from empty.

    Controllable blocks are exactly the decoder layers that expose an MoE
    token dispatcher. Leading dense layers have no signal of their own and
    are left untouched; with ``mirror_dense_prefix`` they instead follow the
    first MoE block's dynamic scale, so a dense prefix does not train at a
    relatively higher LR than the block right after it.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = DynamicBlockLRRescalerConfig

    def __init__(self, config: Config, *, monitoring: MonitoringCallback) -> None:
        self.config = config
        self._monitoring = monitoring
        self._dispatcher: DispatcherSource | None = None
        # block index -> [(param group, its scheduler, group index)] for the
        # groups holding exactly that layer
        self._block_groups: dict[int, list[tuple[dict, Any, int]]] = {}
        # MoE unit name (e.g. "layers.29.moe") -> block index
        self._tag_to_block: dict[str, int] = {}
        # blocks we can control, in a stable order for the decision collective
        self._controllable_blocks: list[int] = []
        # leading dense blocks mirroring the first MoE block's dynamic scale
        self._mirror_blocks: list[int] = []
        # block index -> rolling window of recent max-violation samples
        self._max_vio_history: dict[int, deque[float]] = {}
        # block index -> cumulative dynamic LR scale (1.0 = untouched)
        self._block_scale: dict[int, float] = {}
        # block index -> its base LRs with no dynamic scale, positionally matching
        # _block_groups[block]. Checkpointed: it cannot be re-read from the
        # scheduler after a resume (see _apply_scale).
        self._unscaled_base_lrs: dict[int, list[float]] = {}
        # block index -> capture count at the block's last fire (cooldown)
        self._block_last_fire: dict[int, int] = {}
        self._capture_count = 0
        # Checkpoint state seen before fit_start; re-applied there.
        self._loaded_state: dict[str, Any] = {}
        # True once load_state_dict has read a checkpoint, which is what
        # tells fit_start whether scheduler.base_lrs are still unscaled.
        self._restored = False
        self._pp_mesh = None

    def fit_start(self, state: MutableState) -> None:
        self._pp_mesh = state.trainer.parallel_dims.get_optional_mesh("pp")
        owners = _block_indices(state.trainer.model_parts)
        self._tag_to_block = {}
        for unit in iter_moe_monitor_units(state.trainer.model_parts):
            if unit.dispatcher is None:
                continue
            # Membership by module identity, not by name: the unit's MoE is a
            # submodule of exactly one decoder block.
            block = owners.get(id(unit.module))
            if block is not None:
                self._tag_to_block[unit.name] = block

        self._block_groups = self._map_blocks_to_param_groups(state, owners)
        self._controllable_blocks = sorted(
            set(self._tag_to_block.values()) & set(self._block_groups)
        )
        if not self._controllable_blocks:
            self._enter_noop_or_raise(state)
            return

        self._mirror_blocks = []
        if self.config.mirror_dense_prefix:
            first_moe = self._controllable_blocks[0]
            moe_blocks = set(self._tag_to_block.values())
            self._mirror_blocks = sorted(
                block
                for block in self._block_groups
                if block < first_moe and block not in moe_blocks
            )

        loaded = self._loaded_state
        self._max_vio_history = {
            block: deque(
                loaded.get("max_vio_history", {}).get(block, ()),
                maxlen=self.config.window_size,
            )
            for block in self._controllable_blocks
        }
        self._block_scale = {
            block: loaded.get("block_scale", {}).get(block, 1.0)
            for block in self._controllable_blocks + self._mirror_blocks
        }
        # Default allows an immediate first fire once the window fills.
        self._block_last_fire = {
            block: loaded.get("block_last_fire", {}).get(block, -self.config.cooldown)
            for block in self._controllable_blocks
        }
        self._capture_count = loaded.get("capture_count", 0)
        loaded_base_lrs = loaded.get("unscaled_base_lrs", {})
        self._unscaled_base_lrs = {}
        for block, entries in self._block_groups.items():
            if block in loaded_base_lrs:
                self._unscaled_base_lrs[block] = loaded_base_lrs[block]
                continue
            scaled = [scheduler.base_lrs[index] for _, scheduler, index in entries]
            scale = self._block_scale.get(block, 1.0)
            if self._restored and scale not in (0.0, 1.0):
                logger.warning(
                    "DynamicBlockLRRescaler: no unscaled base LR checkpointed "
                    "for block %d; recovering it as base_lrs / %.6g.",
                    block,
                    scale,
                )
                scaled = [lr / scale for lr in scaled]
            self._unscaled_base_lrs[block] = scaled
        # Absolute, so this is a no-op when the scheduler checkpoint did carry it.
        for block, scale in self._block_scale.items():
            if scale != 1.0:
                self._apply_scale(block, scale, factor=1.0)

        logger.info(
            f"DynamicBlockLRRescaler controlling decoder layers "
            f"{self._controllable_blocks}"
            + (f", mirroring onto {self._mirror_blocks}" if self._mirror_blocks else "")
        )

    def _enter_noop_or_raise(self, state: MutableState) -> None:
        """No controllable blocks on this rank: a config error, or a PP stage."""
        pp_mesh = state.trainer.parallel_dims.get_optional_mesh("pp")
        if pp_mesh is None or pp_mesh.size() <= 1:
            raise ValueError(
                "DynamicBlockLRRescaler found no controllable decoder layers: "
                "it needs optimizer param groups holding exactly one decoder "
                "layer each. Configure optimizer.param_groups with one pattern "
                "per layer (see per_layer_param_groups)."
            )
        total = torch.tensor(
            float(len(self._controllable_blocks)), device=state.trainer.device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
        if float(total) == 0.0:
            raise ValueError(
                "DynamicBlockLRRescaler found no controllable decoder layers "
                "on ANY pipeline stage: it needs optimizer param groups "
                "holding exactly one decoder layer each. Configure "
                "optimizer.param_groups with one pattern per layer (see "
                "per_layer_param_groups)."
            )
        logger.warning(
            "DynamicBlockLRRescaler: this pipeline stage holds no MoE "
            "dispatchers; "
            "the rescaler is a no-op on this rank but keeps joining the "
            "cross-stage logging collective."
        )
        self._mirror_blocks = []
        self._max_vio_history = {}
        self._block_scale = {}
        self._block_last_fire = {}
        self._unscaled_base_lrs = {}
        self._capture_count = self._loaded_state.get("capture_count", 0)

    def _map_blocks_to_param_groups(
        self, state: MutableState, owners: dict[int, int]
    ) -> dict[int, list[tuple[dict, Any, int]]]:
        """Map decoder-layer indices to the optimizer groups that hold them.

        Only groups whose every parameter belongs to one decoder layer are
        controllable; a group mixing layers (e.g. the default catch-all
        group) cannot be scaled per block and is skipped. Each group is
        paired with its LR scheduler and group index so a reduction can be
        written into ``base_lrs`` as well. ``owners`` maps parameter ids to
        their block (see ``_block_indices``) — parameters outside the decoder
        blocks (embeddings, lm_head, the MTP head) resolve to None and mark
        their group uncontrollable, exactly as before.
        """
        schedulers = state.trainer.lr_schedulers.schedulers
        optimizers = state.trainer.optimizers.optimizers
        block_groups: dict[int, list[tuple[dict, Any, int]]] = defaultdict(list)
        for optimizer, scheduler in zip(optimizers, schedulers, strict=True):
            for index, group in enumerate(optimizer.param_groups):
                layers = {owners.get(id(p)) for p in group["params"]}
                if len(layers) == 1 and (layer := layers.pop()) is not None:
                    block_groups[layer].append((group, scheduler, index))
        return dict(block_groups)

    def _apply_scale(self, block: int, scale: float, *, factor: float) -> None:
        """Set the block's base LRs to ``scale``; scale its live LR by ``factor``.

        ``base_lrs`` is written absolutely from the unscaled copy, so applying the
        same scale twice is a no-op and a resume can neither double nor drop it.
        ``group["lr"]`` holds the LR the scheduler already computed for this step,
        so it takes the delta instead — ``factor=1.0`` leaves it alone on resume,
        where the optimizer checkpoint restored it and the next
        ``scheduler.step()`` recomputes it from ``base_lrs`` anyway.
        """
        # strict: a length mismatch means optimizer.param_groups changed shape
        # across a resume, which would silently mis-apply the scales.
        for (group, scheduler, index), base_lr in zip(
            self._block_groups[block], self._unscaled_base_lrs[block], strict=True
        ):
            scheduler.base_lrs[index] = base_lr * scale
            group["lr"] *= factor

    def batch_end(self, state: MutableState) -> None:
        source = self._dispatcher_source()
        if not source.active:
            return

        if self._controllable_blocks:
            view = ObserverState.from_mutable(state)
            for unit_name, values in source.max_vio_by_unit(view).items():
                value = values.get(self.config.max_vio_series)
                if value is None:
                    continue
                block = self._tag_to_block.get(unit_name)
                if block in self._max_vio_history:
                    self._max_vio_history[block].append(float(value))
            self._capture_count += 1

            if state.step >= self.config.warmup_steps:
                self._maybe_reduce_block_lrs(state)
        self._log_block_scales(state)

    def _maybe_reduce_block_lrs(self, state: MutableState) -> None:
        decisions = torch.zeros(
            len(self._controllable_blocks),
            dtype=torch.int32,
            device=state.trainer.device,
        )
        for i, block in enumerate(self._controllable_blocks):
            history = self._max_vio_history[block]
            if len(history) < self.config.window_size:
                continue
            if (
                self._capture_count - self._block_last_fire[block]
                < self.config.cooldown
            ):
                continue
            window = torch.tensor(list(history), dtype=torch.float32)
            if self.should_reduce_block_lr(window):
                decisions[i] = 1

        if dist.is_available() and dist.is_initialized():
            for mesh_name in ("loss", "tp", "ep"):
                mesh = state.trainer.parallel_dims.get_optional_mesh(mesh_name)
                if mesh is not None and mesh.size() > 1:
                    dist.all_reduce(
                        decisions, op=dist.ReduceOp.MAX, group=mesh.get_group()
                    )

        reduced = []
        for i, block in enumerate(self._controllable_blocks):
            if decisions[i].item() != 1:
                continue
            # Record the fire even when the floor blocks the reduction, so the
            # cooldown keeps throttling re-evaluation at the floor.
            self._block_last_fire[block] = self._capture_count
            new_scale = max(
                self._block_scale[block] * self.config.reduce_coef,
                self.config.lr_scaling_floor,
            )
            if new_scale < self._block_scale[block]:
                self._apply_scale(
                    block, new_scale, factor=new_scale / self._block_scale[block]
                )
                self._block_scale[block] = new_scale
                reduced.append(block)

        if self._mirror_blocks:
            source_scale = self._block_scale[self._controllable_blocks[0]]
            for block in self._mirror_blocks:
                current = self._block_scale.get(block, 1.0)
                if source_scale != current:
                    self._apply_scale(
                        block, source_scale, factor=source_scale / current
                    )
                    self._block_scale[block] = source_scale

        if reduced:
            new_scales = {b: round(self._block_scale[b], 4) for b in reduced}
            logger.info(f"DynamicBlockLRRescaler reduced LR for blocks {new_scales}")

    def should_reduce_block_lr(self, max_vios_window: torch.Tensor) -> bool:
        """True when the window's robust relative trend exceeds the dead-band.

        Theil-Sen (median of pairwise slopes) instead of least squares: the
        max-violation signal is spiky, and the median slope shrugs off
        outliers (~29% breakdown point). Normalizing by the window median
        makes the threshold scale-free across layers.
        """
        window = max_vios_window.flatten().float()
        median = window.median().clamp_min(1e-6)
        return (
            self._theil_sen_slope(window) / median.item() > self.config.slope_threshold
        )

    @staticmethod
    def _theil_sen_slope(y: torch.Tensor) -> float:
        """Theil-Sen slope over x = linspace(0, 1, n).

        The unit x-range makes the magnitude independent of the window length
        (the sign, hence the decision, is unaffected by the x-scaling).
        """
        y = y.flatten().float()
        n = y.numel()
        if n < 2:
            return 0.0
        upper = torch.triu_indices(n, n, offset=1, device=y.device)
        dy = y[upper[1]] - y[upper[0]]
        x = torch.linspace(0.0, 1.0, n, device=y.device, dtype=y.dtype)
        dx = x[upper[1]] - x[upper[0]]
        return torch.median(dy / dx).item()

    def _log_block_scales(self, state: MutableState) -> None:
        """Continuous per-block series of the cumulative dynamic scale.

        Only the scale is logged here — the applied LR already lives in the
        groups and ``base_lrs``, so TrainerSource's per-group lr series reports it.
        Under pipeline parallelism each stage controls its own blocks, so the
        per-stage tags are gathered to the logging stage.
        """
        metrics = {
            f"dynamic_block_lr/scale/block_{block}": self._block_scale.get(block, 1.0)
            for block in self._controllable_blocks + self._mirror_blocks
        }
        state.record_all(metrics)

    def _dispatcher_source(self) -> DispatcherSource:
        """The dispatcher source this callback steers from."""
        if self._dispatcher is None:
            source = self._monitoring.find_source(DispatcherSource)
            if source is None:
                raise ValueError(
                    "DynamicBlockLRRescaler steers from the dispatcher "
                    "max-violation, which is produced by DispatcherSource. "
                    "List 'dispatcher' in callbacks.monitoring.sources, or "
                    "drop this callback from callbacks."
                )
            published = [variant.name for variant in source.config.max_vio]
            if self.config.max_vio_series not in published:
                raise ValueError(
                    f"DynamicBlockLRRescaler steers from "
                    f"'{self.config.max_vio_series}', which DispatcherSource "
                    f"does not publish. Its max_vio series are {published}. "
                    "Add it there, or point max_vio_series at one of them."
                )
            self._dispatcher = source
        return self._dispatcher

    def stateful_objects(self) -> dict[str, Any]:
        return {"dynamic_block_lr_rescaler": self}

    def state_dict(self) -> dict[str, Any]:
        payload = {
            "block_scale": {
                str(block): scale for block, scale in self._block_scale.items()
            },
            "unscaled_base_lrs": {
                str(block): list(base_lrs)
                for block, base_lrs in self._unscaled_base_lrs.items()
            },
            "max_vio_history": {
                str(block): list(history)
                for block, history in self._max_vio_history.items()
            },
            "block_last_fire": {
                str(block): count for block, count in self._block_last_fire.items()
            },
            "capture_count": self._capture_count,
        }

        if self._pp_mesh is not None and self._pp_mesh.size() > 1:
            group = self._pp_mesh.get_group()
            gathered: list[dict[str, Any] | None] = [None] * self._pp_mesh.size()
            dist.all_gather_object(gathered, payload, group=group)
            merged: dict[str, Any] = {
                key: {} for key in payload if key != "capture_count"
            }
            merged["capture_count"] = 0
            for part in gathered:
                assert part is not None
                for key, value in part.items():
                    if key == "capture_count":
                        merged[key] = max(merged[key], value)
                    else:
                        merged[key].update(value)
            payload = merged
        return {"rescaler_state": json.dumps(payload)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        payload = state_dict.get("rescaler_state")
        loaded = json.loads(payload) if payload else {}
        self._loaded_state = {
            "block_scale": {
                int(block): float(scale)
                for block, scale in loaded.get("block_scale", {}).items()
            },
            "unscaled_base_lrs": {
                int(block): [float(lr) for lr in base_lrs]
                for block, base_lrs in loaded.get("unscaled_base_lrs", {}).items()
            },
            "max_vio_history": {
                int(block): [float(v) for v in history]
                for block, history in loaded.get("max_vio_history", {}).items()
            },
            "block_last_fire": {
                int(block): int(count)
                for block, count in loaded.get("block_last_fire", {}).items()
            },
            "capture_count": int(loaded.get("capture_count", 0)),
        }
        # fit_start must know a checkpoint was read: without this it cannot tell
        # a fresh run (where scheduler.base_lrs really are unscaled) from a
        # resumed one (where they already carry a previous run's scale).
        self._restored = bool(loaded)


def per_layer_param_groups(num_layers: int) -> list[ParamGroupConfig]:
    """One optimizer param group per decoder layer.

    Feed into ``OptimizersContainer.Config.param_groups`` so every decoder
    layer lands in its own group — the granularity DynamicBlockLRRescaler
    controls at. All multipliers stay 1.0: the grouping changes nothing until
    the rescaler (or an explicit static multiplier) acts on it.
    """
    return [
        ParamGroupConfig(pattern=rf"(^|\.)layers\.{layer}\.")
        for layer in range(num_layers)
    ]
