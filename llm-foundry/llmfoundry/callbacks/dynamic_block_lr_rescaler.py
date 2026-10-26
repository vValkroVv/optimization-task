# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Dynamic per-block learning rate control driven by MoE token-dispatch stats."""

import logging
import re
from collections import defaultdict, deque
from typing import Dict, List, Optional

import torch

from composer.core import State
from composer.loggers import Logger
from composer.utils import dist

from llmfoundry.callbacks.token_dispatcher_monitor import TokenDispatcherMonitor

__all__ = ["DynamicBlockLRRescaler"]

log = logging.getLogger(__name__)

# Matches the decoder layer index in a parameter name, e.g. "...layers.29.mlp...".
_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")
# Matches the MoE-layer name tag produced by TokenDispatcherMonitor, e.g. "layer_29".
_LAYER_TAG_RE = re.compile(r"^layer_(\d+)$")


class DynamicBlockLRRescaler(TokenDispatcherMonitor):
    """Reduce the learning rate of individual decoder blocks on the fly.

    Builds on :class:`TokenDispatcherMonitor`: it reuses the per-layer MoE
    max-violation statistics computed there, accumulates them over a rolling
    window of recent logging steps, and periodically asks a heuristic whether a
    given block's learning rate should be reduced. The heuristic fits a robust
    Theil-Sen slope to the block's recent max-violation window and fires when the
    relative trend exceeds ``slope_threshold``. When it fires for block ``i``,
    every parameter of decoder layer ``i`` (attention + MoE) has its effective LR
    multiplied by ``reduce_coef``.

    Mechanism: the reduction is applied via a **per-parameter LR multiplier** on
    the optimizer (``optimizer.set_lr_scale(param, scale)``), not via optimizer
    param groups. This means the user only has to enable the callback -- no
    per-block ``param_groups`` config is required, no extra optimizer groups are
    created, and the scale composes multiplicatively with whatever LR the
    scheduler sets each step (so no scheduler ``base_lrs`` bookkeeping). The
    optimizer must expose ``set_lr_scale`` (the ``DecoupledAdamW`` family does);
    otherwise the callback logs a warning and stays inert.

    Composition with a static config base: this callback owns only the *dynamic*
    multiplier (``set_lr_scale``). A static, config-driven base scale set via the
    optimizer's ``lr_scales`` (stored as ``base_lr_scale``) is read separately
    from the optimizer at every ``fit_start`` and the optimizer applies the
    product ``base * dynamic`` each step. So a block configured with base 0.5
    starts at 0.5 and any dynamic reduction stacks on top; the logged
    ``scale/block_{i}`` reports the effective ``base * dynamic``. The base is
    re-read on resume (config changes take effect) and is NOT checkpointed.

    The dynamic per-block reduction starts at 1.0 and is clamped so it never drops
    below ``lr_scaling_floor`` (relative to the block's base), so the effective LR
    bottoms out at ``base * lr_scaling_floor`` times the original LR. No rescaling
    happens before ``warmup_steps`` batches (to skip the noisy early phase of
    training). The dynamic scale is checkpointed and re-applied on resume.

    Controllable blocks are exactly the decoder layers that expose an MoE token
    dispatcher (the source of the max-violation signal); every such layer is
    controlled automatically. Leading dense layers (the first ``k`` decoder
    layers, which have no MoE dispatcher and thus no signal of their own) are
    normally left untouched; with ``mirror_dense_prefix=True`` they instead follow
    the first MoE block -- their dynamic scale is copied from it (their own static
    base is preserved), so a dense prefix does not get a relatively higher LR than
    the block right after it.

    Args:
        reduce_coef (float): Multiplicative factor applied to a block's effective
            LR (all of layer ``i``'s params) when the heuristic fires. Defaults
            to 0.99.
        window_size (int): Number of recent per-layer max-violation samples in
            the trailing window the heuristic fits its slope to. One sample is
            produced every ``batch_log_interval`` batches. The window is
            checkpointed and restored on resume, so it keeps filling across
            restarts. Defaults to 100.
        cooldown (int): Minimum number of samples between consecutive reductions
            of the same block. The trailing window is evaluated on every sample,
            but a block can fire at most once per ``cooldown`` samples (so a
            sustained rise does not collapse the LR every step). Defaults to 20;
            pass ``None`` to use ``window_size`` (one full window between fires).
        slope_threshold (float): Relative-slope dead-band ``tau``. A block fires
            only when ``theil_sen_slope(window) / median(window) > slope_threshold``
            -- i.e. when the robust trend predicts a rise of more than
            ``slope_threshold`` of the window's level. Normalizing by the median
            makes this scale-free across layers; the dead-band suppresses firing
            on noise around a flat trend. ``0.0`` reproduces a bare ``slope > 0``
            test. Defaults to 0.05 (tuned on recorded max-violation traces).
        lr_scaling_floor (float): Hard lower bound on the cumulative per-block LR
            scale, relative to the block's original LR. The LR is never reduced
            below ``lr_scaling_floor * original_lr``. Defaults to 0.3.
        warmup_steps (int): No rescaling is performed before this many batches
            have elapsed. Must be greater than ``window_size``. Defaults to 2000.
        mirror_dense_prefix (bool): When ``True``, the leading dense decoder layers
            (index below the first MoE block) mirror the first MoE block's dynamic
            LR scale instead of being left at 1.0. Each mirrored layer keeps its
            own static base, so its effective LR is ``base * dynamic_first_moe``.
            Only the contiguous dense prefix is mirrored; dense layers interleaved
            above the first MoE block are untouched. Defaults to ``False``.
    """

    def __init__(
        self,
        reduce_coef: float = 0.99,
        window_size: int = 100,
        cooldown: Optional[int] = 20,
        slope_threshold: float = 0.05,
        lr_scaling_floor: float = 0.3,
        warmup_steps: int = 2000,
        mirror_dense_prefix: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert 0.0 < reduce_coef <= 1.0, f"reduce_coef must be in (0, 1]; got {reduce_coef}"
        assert 0.0 < lr_scaling_floor <= 1.0, (
            f"lr_scaling_floor must be in (0, 1]; got {lr_scaling_floor}"
        )
        assert slope_threshold >= 0.0, f"slope_threshold must be >= 0; got {slope_threshold}"
        # Ensures warmup spans at least one full evaluation window, so the first
        # decision is never made on (partly) pre-warmup chaotic statistics.
        assert warmup_steps > window_size, (
            f"warmup_steps ({warmup_steps}) must be greater than window_size ({window_size})"
        )
        self.reduce_coef = reduce_coef
        self.window_size = window_size
        self.cooldown = cooldown if cooldown is not None else window_size
        self.slope_threshold = slope_threshold
        self.lr_scaling_floor = lr_scaling_floor
        self.warmup_steps = warmup_steps
        self.mirror_dense_prefix = mirror_dense_prefix

        # Leading dense decoder layers (index < first controllable MoE block) that
        # have no max-violation signal of their own. When ``mirror_dense_prefix``
        # is set, each mirrors the DYNAMIC scale of the first MoE block (its own
        # static base is preserved); empty otherwise.
        self._mirror_blocks: List[int] = []
        # block index -> rolling window of recent max-violation samples
        self._max_vio_history: Dict[int, deque] = {}
        # block index -> the trainable parameters of that decoder layer
        self._block_to_params: Dict[int, List[torch.Tensor]] = {}
        # MoE name tag (e.g. "layer_29") -> block index
        self._tag_to_block: Dict[str, int] = {}
        # blocks we can actually control, in a stable order for collective ops
        self._controllable_blocks: List[int] = []
        # block index -> cumulative DYNAMIC LR scale applied by this callback
        # (starts at 1.0, reduced toward lr_scaling_floor). This is the part we
        # checkpoint; it is multiplied by the static base scale below to get the
        # effective multiplier the optimizer applies.
        self._block_scale: Dict[int, float] = {}
        # block index -> STATIC base LR scale from the optimizer config
        # (``lr_scales``); 1.0 unless the config set it. Re-read from the optimizer
        # at every fit_start (so a config change is picked up on resume) and NOT
        # checkpointed. Effective scale = base * dynamic.
        self._block_base_scale: Dict[int, float] = {}
        # block index -> capture_count at which the block last fired (for cooldown)
        self._block_last_fire: Dict[int, int] = {}
        # State restored from a checkpoint, re-applied in _fit_start. The rolling
        # window, capture counter and last-fire markers MUST survive restarts:
        # otherwise every resume wipes the window and it can never refill the
        # window_size samples needed to fire (the original failure mode on the
        # frequently-restarting 700B run).
        self._loaded_block_scale: Dict[int, float] = {}
        self._loaded_max_vio_history: Dict[int, List[float]] = {}
        self._loaded_block_last_fire: Dict[int, int] = {}
        self._loaded_capture_count = 0
        self._capture_count = 0

    def _fit_start(self, state: State, logger: Logger):
        # Parent populates self.moe_layer_dispatchers / self.name_tags here, and
        # this is invoked from both fit_start and after_load (resume).
        super()._fit_start(state, logger)

        self._build_tag_to_block()
        self._build_block_params(state)

        # A block is controllable if it has an MoE tag (the max-violation signal)
        # and at least one trainable parameter we can scale.
        self._controllable_blocks = sorted(
            block
            for block in set(self._tag_to_block.values())
            if self._block_to_params.get(block)
        )

        # The per-parameter LR scale lives on the optimizer; bail out (inert) if
        # the configured optimizer does not support it.
        optimizer = state.optimizers[0]
        self._optimizer_supports_scaling = hasattr(optimizer, "set_lr_scale")
        if not self._optimizer_supports_scaling:
            log.warning(
                "DynamicBlockLRRescaler is enabled but optimizer "
                f"{type(optimizer).__name__} has no `set_lr_scale`; no per-block "
                "LR control will be applied. Use an optimizer that supports "
                "per-parameter LR scaling (e.g. DecoupledAdamW)."
            )
            self._controllable_blocks = []

        # Leading dense blocks to couple to the first MoE block's dynamic scale.
        # Dense = has trainable params but no MoE max-violation tag; "leading" =
        # index below the first controllable block (the typical "first k layers
        # are dense" case). Interleaved dense layers above the first MoE block are
        # left untouched.
        self._mirror_blocks = []
        if self.mirror_dense_prefix and self._controllable_blocks:
            first_moe = self._controllable_blocks[0]
            moe_blocks = set(self._tag_to_block.values())
            self._mirror_blocks = sorted(
                block
                for block in self._block_to_params
                if block < first_moe and block not in moe_blocks
            )

        # Restore the rolling window from checkpoint if present, else start empty.
        # The deque is bounded to the (possibly reconfigured) window_size; if the
        # window was shrunk since the checkpoint, deque keeps the most recent
        # samples automatically.
        self._max_vio_history = {
            block: deque(
                self._loaded_max_vio_history.get(block, ()),
                maxlen=self.window_size,
            )
            for block in self._controllable_blocks
        }
        # Read the STATIC base scale (config `lr_scales`) from the optimizer. This
        # is re-read on every fit_start -- including resumes -- so changing the
        # config base takes effect immediately, and it is intentionally kept out
        # of the checkpoint. The effective multiplier is base * dynamic.
        self._block_base_scale = {
            block: self._optimizer_block_base_scale(optimizer, block)
            for block in self._controllable_blocks + self._mirror_blocks
        }
        # Restore the cumulative DYNAMIC scale from checkpoint if present; else
        # start at 1.0 (no reduction yet). The dynamic part reduces from 1.0 on
        # top of the static base; the floor clamps the dynamic part, so the
        # effective LR bottoms out at base * lr_scaling_floor.
        self._block_scale = {
            block: self._loaded_block_scale.get(block, 1.0)
            for block in self._controllable_blocks
        }
        # Restore the last-fire markers and capture counter so the cooldown keeps
        # throttling across restarts. Default allows an immediate first fire.
        self._block_last_fire = {
            block: self._loaded_block_last_fire.get(block, -self.cooldown)
            for block in self._controllable_blocks
        }
        self._capture_count = self._loaded_capture_count

        # Push the (possibly checkpoint-restored) scales onto the optimizer so a
        # resumed run continues from where it left off.
        for block in self._controllable_blocks:
            self._apply_block_scale(optimizer, block)
        # Couple the leading dense blocks to the first MoE block's scale (no-op
        # unless mirror_dense_prefix is set).
        self._apply_mirror_blocks(optimizer)

        if self._controllable_blocks:
            log.info(
                f"DynamicBlockLRRescaler controlling {len(self._controllable_blocks)} "
                f"decoder layers: {self._controllable_blocks}"
            )
        if self._mirror_blocks:
            log.info(
                f"DynamicBlockLRRescaler mirroring first MoE block "
                f"{self._controllable_blocks[0]}'s scale onto leading dense blocks: "
                f"{self._mirror_blocks}"
            )

    def _build_tag_to_block(self):
        self._tag_to_block = {}
        for tag in self.name_tags:
            match = _LAYER_TAG_RE.match(tag)
            # Main decoder layers only; MTP layers ("mtp_layer_*") are skipped.
            if match is not None:
                self._tag_to_block[tag] = int(match.group(1))

    def _build_block_params(self, state: State):
        """Map each decoder-layer index to its trainable parameters (whole layer)."""
        block_to_params: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for name, param in state.model.named_parameters():
            if not param.requires_grad:
                continue
            # MTP params (e.g. "mtp_block.mtp_layers.j.decoder_layer.layers.0...")
            # also contain ".layers.N." but belong to the MTP head, not main
            # decoder layer N. Exclude them so block N's scale doesn't leak there.
            if "mtp" in name:
                continue
            match = _LAYER_INDEX_RE.search(name)
            if match is not None:
                block_to_params[int(match.group(1))].append(param)
        self._block_to_params = dict(block_to_params)

    def _apply_block_scale(self, optimizer, block: int):
        """Push the block's current DYNAMIC scale onto every param of layer ``block``.

        Only the dynamic part is written here (via ``set_lr_scale``); the optimizer
        multiplies it by the static base (``base_lr_scale``) internally, so the
        effective multiplier is ``base * dynamic`` without this callback ever
        touching the configured base.
        """
        scale = self._block_scale[block]
        for param in self._block_to_params[block]:
            optimizer.set_lr_scale(param, scale)

    def _apply_mirror_blocks(self, optimizer):
        """Copy the first MoE block's DYNAMIC scale onto the leading dense blocks.

        Only the dynamic part is mirrored (via ``set_lr_scale``); each dense
        block keeps its own static base, so its effective LR is
        ``base_dense * dynamic_first_moe``. Idempotent and cheap; safe to call
        every step. No-op when ``mirror_dense_prefix`` is off or there are no
        leading dense blocks.
        """
        if not self._mirror_blocks:
            return
        source = self._controllable_blocks[0]
        scale = self._block_scale[source]
        for block in self._mirror_blocks:
            # Track for logging; the optimizer multiplies by the dense base.
            self._block_scale[block] = scale
            for param in self._block_to_params[block]:
                optimizer.set_lr_scale(param, scale)

    def _optimizer_block_base_scale(self, optimizer, block: int) -> float:
        """Read the block's STATIC base LR scale (config ``lr_scales``) from the optimizer.

        Returns the minimum base scale across the block's params (the most-reduced
        one) so a uniformly-configured block is reported exactly; defaults to 1.0
        if the block has no params or the optimizer exposes no base-scale getter.
        """
        params = self._block_to_params.get(block, [])
        getter = getattr(optimizer, "get_base_lr_scale", None)
        if not params or getter is None:
            return 1.0
        return min(getter(param) for param in params)

    def should_reduce_block_lr(self, max_vios_window: torch.Tensor) -> bool:
        """Decide whether a block's learning rate should be reduced.

        Fits a robust Theil-Sen slope (median of all pairwise slopes) to the
        block's recent max-violation samples, normalizes it by the window's
        median level (making the test scale-free across layers), and reduces the
        LR when that relative trend exceeds ``slope_threshold``. Theil-Sen is used
        instead of a least-squares fit because the max-violation signal is noisy
        and spiky; the median of pairwise slopes is insensitive to outliers (~29%
        breakdown point). The ``slope_threshold`` dead-band prevents firing on
        noise wobbling above zero around an otherwise flat trend.

        Args:
            max_vios_window (torch.Tensor): 1D tensor of the block's most recent
                per-layer max-violation samples (oldest first), covering up to
                ``window_size`` logging steps. Values are already all-reduced
                across ranks by the parent monitor.

        Returns:
            bool: ``True`` if the median-normalized slope exceeds
                ``slope_threshold``.
        """
        window = max_vios_window.flatten().float()
        median = window.median().clamp_min(1e-6)
        relative_slope = self._theil_sen_slope(window) / median.item()
        return relative_slope > self.slope_threshold

    @staticmethod
    def _theil_sen_slope(y: torch.Tensor) -> float:
        """Theil-Sen slope estimate over x = linspace(0, 1, n).

        Using a unit x-range makes the slope magnitude independent of the window
        length (the sign, hence the decision, is unaffected by the x-scaling).
        """
        y = y.flatten().float()
        n = y.numel()
        if n < 2:
            return 0.0
        x = torch.linspace(0.0, 1.0, n, device=y.device, dtype=y.dtype)
        # Upper-triangular pairs (i < j) so every dx = x[j] - x[i] is positive.
        iu = torch.triu_indices(n, n, offset=1, device=y.device)
        dy = y[iu[1]] - y[iu[0]]
        dx = x[iu[1]] - x[iu[0]]
        return torch.median(dy / dx).item()

    def batch_end(self, state: State, logger: Logger):
        super().batch_end(state, logger)
        # The parent only refreshes its statistics on its logging interval.
        if state.timestamp.batch.value % self.batch_log_interval != 0:
            return

        self._capture_max_vios()
        self._capture_count += 1

        # Skip the noisy early phase of training entirely.
        if state.timestamp.batch.value >= self.warmup_steps and self._controllable_blocks:
            self._maybe_reduce_block_lrs(state, logger)

        # Always log the current per-block LR / scale (every interval, including
        # warmup) so each block is a continuous, plottable time series.
        if self._controllable_blocks:
            self._log_block_lrs(state, logger)

    def _capture_max_vios(self):
        for tag, value in self._latest_per_layer_max_vio.items():
            block = self._tag_to_block.get(tag)
            if block in self._max_vio_history:
                # Store as a plain Python float so the rolling window is trivially
                # checkpointable (no CUDA tensors in the callback state_dict).
                self._max_vio_history[block].append(float(value))

    def _maybe_reduce_block_lrs(self, state: State, logger: Logger):
        device = f"cuda:{torch.cuda.current_device()}"
        decisions = torch.zeros(
            len(self._controllable_blocks), dtype=torch.int32, device=device
        )
        for i, block in enumerate(self._controllable_blocks):
            history = self._max_vio_history[block]
            # Need a full window, and the cooldown since the last fire must have
            # elapsed, before this block may fire again.
            if len(history) < self.window_size:
                continue
            if self._capture_count - self._block_last_fire[block] < self.cooldown:
                continue
            window = torch.tensor(list(history), dtype=torch.float32)
            if self.should_reduce_block_lr(window):
                decisions[i] = 1

        # Ensure every rank applies the exact same LR changes. Even though the
        # max-violation inputs are all-reduced, we guard against any rank-local
        # divergence in the heuristic by agreeing on the decision (MAX = "if any
        # rank votes to reduce, all reduce").
        if dist.get_world_size() > 1:
            dist.all_reduce(decisions, reduce_operation="MAX")

        optimizer = state.optimizers[0]
        reduced_blocks: List[int] = []
        for i, block in enumerate(self._controllable_blocks):
            # Record the fire even if the floor blocks the actual reduction, so
            # the cooldown still throttles re-evaluation at the floor.
            if decisions[i].item() != 1:
                continue
            self._block_last_fire[block] = self._capture_count
            # Clamp the cumulative scale to the floor. The actual new scale may
            # land above ``reduce_coef * prev`` on the last step before the floor,
            # or equal ``prev`` (no-op) once the block already sits at the floor.
            new_scale = max(self._block_scale[block] * self.reduce_coef, self.lr_scaling_floor)
            if new_scale >= self._block_scale[block]:  # already at the floor
                continue
            self._block_scale[block] = new_scale
            self._apply_block_scale(optimizer, block)
            reduced_blocks.append(block)

        # Re-mirror so the leading dense blocks follow any change to the first MoE
        # block's scale this step (no-op unless mirror_dense_prefix is set).
        if reduced_blocks and self._mirror_blocks:
            self._apply_mirror_blocks(optimizer)

        if reduced_blocks:
            log.info(
                "DynamicBlockLRRescaler reduced LR for blocks "
                f"{reduced_blocks} (cumulative scales: "
                f"{ {b: round(self._block_scale[b], 4) for b in reduced_blocks} })"
            )

    def _log_block_lrs(self, state: State, logger: Logger):
        """Log every controllable block's current scale and effective LR.

        Emitted every logging interval (not only on changes) so each block is a
        continuous time series. ``dynamic_block_lr/scale/block_{i}`` is the
        EFFECTIVE multiplier ``base * dynamic`` (so a config ``lr_scales`` base of
        e.g. 0.5 shows as the 0.5 baseline, with the dynamic reduction stacking on
        top, bottoming out at ``base * lr_scaling_floor``) and
        ``dynamic_block_lr/lr/block_{i}`` is the effective LR (scheduler LR x that
        multiplier). Mirrored leading-dense blocks are logged the same way (their
        dynamic part tracks the first MoE block).
        """
        base_lr = state.optimizers[0].param_groups[0]["lr"]
        metrics = {}
        for block in self._controllable_blocks + self._mirror_blocks:
            effective_scale = self._block_base_scale[block] * self._block_scale[block]
            metrics[f"dynamic_block_lr/scale/block_{block}"] = effective_scale
            metrics[f"dynamic_block_lr/lr/block_{block}"] = base_lr * effective_scale
        logger.log_metrics(metrics)

    def state_dict(self) -> dict:
        # Persist:
        #  - cumulative scales, so the floor stays relative to the true original
        #    LR after a resume (the optimizer restores the reduced LRs);
        #  - the rolling max-violation window, capture counter and last-fire
        #    markers, so a resumed run keeps accumulating toward window_size and
        #    keeps throttling via the cooldown instead of restarting from empty.
        return {
            # Only the controllable blocks carry real dynamic state; mirrored
            # dense-prefix blocks are derived from the first MoE block each
            # fit_start, so they are not persisted.
            "block_scale": {
                block: self._block_scale[block]
                for block in self._controllable_blocks
            },
            "max_vio_history": {
                block: list(history)
                for block, history in self._max_vio_history.items()
            },
            "block_last_fire": dict(self._block_last_fire),
            "capture_count": self._capture_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self._loaded_block_scale = dict(state.get("block_scale", {}))
        self._loaded_max_vio_history = {
            block: list(history)
            for block, history in state.get("max_vio_history", {}).items()
        }
        self._loaded_block_last_fire = dict(state.get("block_last_fire", {}))
        self._loaded_capture_count = int(state.get("capture_count", 0))
