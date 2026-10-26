# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Dynamic per-block MoE bias-update-rate control driven by token-dispatch stats.

Two-stage design:

1. **Estimate the true signal from noisy activations.** The per-layer
   max-violation samples produced by :class:`TokenDispatcherMonitor` are a noisy,
   spike-prone observation of the block's *true* imbalance. Each block runs a
   robust local-linear-trend Kalman filter (:class:`_LocalTrendKF`) that turns the
   raw stream into a smoothed estimate of the true level **and its slope**, while
   rejecting outlier spikes (e.g. an lr-explosion transient) instead of reacting to
   them.

2. **Rescale the loss-free bias rate to keep max-violation non-increasing.** A
   one-way ratchet damps the block's bias-update-rate scale whenever the estimated
   level rises above the lowest level it has reached since warmup (a running-min
   envelope, beyond a dead-band), and holds it otherwise -- it never raises the gain
   on its own. Because the gain is monotone non-increasing it cannot oscillate, and
   because the envelope ratchets down it keeps max-violation at or below its best-so-
   far while leaving already-well-behaved blocks untouched. All decisions are on
   ratios, so the same knobs hold across model sizes.
"""

import logging
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import torch

from composer.core import State
from composer.loggers import Logger
from composer.utils import dist

from llmfoundry.callbacks.token_dispatcher_monitor import TokenDispatcherMonitor
from llmfoundry.callbacks.dynamic_block_lr_rescaler import DynamicBlockLRRescaler
from llmfoundry.callbacks._trend_kalman import _LocalTrendKF

try:  # optional: the algorithm we drive lives in the composer contrib submodule
    from composer.algorithms.update_bias.update_bias import UpdateBias
except Exception:  # pragma: no cover - composer build without UpdateBias
    UpdateBias = None

__all__ = ["DynamicBlockBiasRateRescaler"]

log = logging.getLogger(__name__)

# Matches the MoE-layer name tag produced by TokenDispatcherMonitor, e.g. "layer_29".
# The index is the REAL decoder-layer index (enumerate over model.layers), the same
# convention UpdateBias uses to key its per-block scales.
_LAYER_TAG_RE = re.compile(r"^layer_(\d+)$")
# Matches the decoder layer index in a parameter name, e.g. "...layers.29.mlp...".
_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


class DynamicBlockBiasRateRescaler(TokenDispatcherMonitor):
    """Dynamically damp the loss-free-balancing bias-update rate of growing blocks.

    Sibling of :class:`DynamicBlockLRRescaler`, but for the DeepSeek-style
    auxiliary-loss-free load-balancing gain ``gamma`` (``moe_router_bias_update_rate``)
    rather than the learning rate. It consumes the per-layer MoE max-violation signal
    from :class:`TokenDispatcherMonitor` and runs a **slow, robust local-linear-trend
    Kalman filter** per block (:class:`_LocalTrendKF`) to recover the *true* level from
    the noisy, spike-prone stream. The controller then damps a block whenever its level
    rises above the **lowest level it has reached since warmup** (a running-min envelope
    that ratchets down, locking in any improvement).

    **Why this design (all choices validated against real ``max_vio`` traces + a plant
    identified from a controlled run).**

    * *Level, not slope.* The per-sample slope is dominated by noise (a flat block spends
      ~half its samples with a positive slope estimate), so a slope-driven controller
      rectifies that noise and drives *every* block to the floor, discriminating nothing.
      The *level relative to a reference* is what actually separates a growing block from
      a stable one.
    * *Running-min reference.* Using the lowest level so far makes the target ratchet
      down as a block improves, so the controller keeps max-violation at or below its
      best-so-far -- "non-increasing" in the strict sense the reference tracks.
    * *One-way ratchet (no recovery by default).* The identified plant is weak and
      self-correcting, so oscillation is not plant resonance -- it comes from a controller
      that both lowers and raises the gain (a leak-vs-damp limit cycle). A gain that only
      ever decreases or holds simply cannot oscillate, and on the estimated plant it still
      drives an ever-rising block's max-violation monotonically down.

    **Control law (per sample, per block).** With ``L`` the KF level, ``env`` the running
    minimum of ``L`` since warmup, and ``excess = (L - env) / max(env, eps)`` the fraction
    by which the level currently sits above that envelope::

        env <- min(env, L)                                     # ratchet the reference down
        if excess > rise_tolerance:                            # rose too far -> damp
            cut   <- min(max_damp_per_step, _DAMP_GAIN * (excess - rise_tolerance))
            scale <- max(scale_floor, scale * (1 - cut))
        elif allow_recovery and excess < recover_tolerance and slope <= 0:
            scale <- min(recovery_ceiling, scale * (1 + _RECOVERY_STEP))   # reclaim gain
        else:
            scale <- scale                                     # hold

    Every knob is either a **fraction of the level** (``excess``, ``rise_tolerance``,
    ``recover_tolerance``) or a **pure gain multiplier in [0, 1]** (``scale_floor``,
    ``recovery_ceiling``, ``max_damp_per_step``); nothing references an absolute
    max-violation value, so the same numbers work at any baseline imbalance -- i.e.
    **across model sizes** -- with no per-run retuning.

    With the default ``allow_recovery=False`` the gain is a strict **one-way ratchet**
    (decrease-or-hold): non-increasing, non-oscillating, and it damps only the blocks that
    rise more than ``rise_tolerance`` above their envelope -- proportionally to how far --
    while leaving stable/improving blocks untouched.

    ``allow_recovery=True`` reclaims loss-free gain, but only while a block is *hugging its
    envelope* (``excess < recover_tolerance``, a one-sided dead-band below the damp band)
    and *not rising* (KF slope ``<= 0``), easing back up toward ``recovery_ceiling``. The
    hold zone between ``recover_tolerance`` and ``rise_tolerance`` buffers the recovery so
    damping does not immediately re-trigger; recovery still ripples during genuine
    improve/rise transitions, so the default leaves it off.

    **Mechanism.** The scale is applied via a per-block DYNAMIC multiplier on the
    :class:`UpdateBias` algorithm (``update_bias.set_bias_rate_scale(block, scale)``),
    mirroring how the LR rescaler pokes ``optimizer.set_lr_scale``. A static,
    config-driven base (``bias_update_rate_scales``) is read separately; the effective
    rate is ``global_rate * base * dynamic``. The callback stays inert (logs a warning)
    if no ``UpdateBias`` algorithm with the scale API is present.

    **History reuse / seeding (works from existing checkpoints).** A freshly-added
    callback is absent from existing checkpoints (composer keys callback state by class
    name), so its window starts empty. Because the max-violation signal is identical to
    the LR rescaler's, state is seeded on resume in this priority order:

    1. our own checkpointed window + KF state + envelope + scale, if present;
    2. else the live sibling :class:`DynamicBlockLRRescaler`'s window (replayed through a
       fresh slow KF to warm the estimator; the envelope is reconstructed as the minimum
       KF level over the replayed window);
    3. else, if this is a *continuation* (``batch > 0``) and ``allow_missing_history``
       is not set, **raise** -- refusing to run blind;
    4. else (fresh run, or opt-out) start empty.

    Args:
        rise_tolerance (float): Fractional rise above the lowest-since-warmup level that is
            tolerated before damping (``0.10`` = damp once a block's max-violation sits
            >10% above its best-so-far). Bigger = more lenient. Defaults to 0.10.
        max_damp_per_step (float): Largest single-step cut to a block's gain, as a fraction
            (``0.15`` = cut by at most 15% per control step). Bigger = damps faster.
            Defaults to 0.15.
        scale_floor (float): Lowest the gain multiplier may fall to (``0.5`` = never below
            0.5x nominal). Defaults to 0.5.
        allow_recovery (bool): Opt-in. If ``True``, reclaim gain while a block hugs its
            envelope and is not rising. Relaxes strict monotonicity (ripples). Defaults to
            ``False`` (strict one-way ratchet).
        recovery_ceiling (float): Highest the gain multiplier may be reclaimed to (only when
            ``allow_recovery``). Kept below 1.0 because the identified neutral scale is
            ~0.85; reclaiming past it re-heats the bias loop. Defaults to 0.85.
        recover_tolerance (float): One-sided dead-band for recovery: only reclaim gain while
            the level is within this fraction of its envelope (only when ``allow_recovery``;
            must be < ``rise_tolerance``). Defaults to 0.03.
        window_size (int): Rolling window of recent raw max-violation samples retained for
            seeding/checkpointing (and to warm the KF before the first action).
            Checkpointed and restored on resume. Defaults to 100.
        warmup_steps (int): No rescaling before this many batches (must exceed
            ``window_size``). Defaults to 5000.
        param_str_match (str): Optional regex restricting which decoder layers this
            callback controls (``re.search`` over trainable param names). ``None``
            (default) controls every MoE block.
        allow_missing_history (bool): If ``True``, a continuation with no own and no
            sibling history starts empty instead of raising. Defaults to ``False``.
        scale_bounds_mode (str): ``"dynamic"`` preserves legacy semantics: ``scale_floor``
            and checkpointed ``block_scale`` bound the dynamic multiplier applied on top of
            static ``bias_update_rate_scales``. ``"effective"`` makes the controller state
            and bounds refer to the effective multiplier, while converting to the dynamic
            multiplier before calling ``UpdateBias``. Defaults to ``"dynamic"``.
        rewrite_checkpoint_stats (dict): Optional config-provided overlay for callback
            state on ``fit_start``. Supports ``block_scale``, ``max_vio_history``,
            ``kf_state`` and ``env`` maps keyed by decoder-layer index. Useful when
            attaching the callback to a checkpoint that was saved without it.
    """

    _DAMP_GAIN = 8.0        # damp slope on fractional overshoot; saturates the cap for any real rise
    _RECOVERY_STEP = 0.01   # per-step gain reclaim fraction while recovering (opt-in)
    # Slow-KF presets (data-fit ratios, scale-free): smooth enough that the running-min
    # envelope tracks the genuine floor, not transient noise troughs.
    KF_LEVEL_RATIO = 0.03
    KF_TREND_RATIO = 0.003
    KF_OBS_HALFLIFE = 60.0
    SCALE_BOUNDS_DYNAMIC = "dynamic"
    SCALE_BOUNDS_EFFECTIVE = "effective"

    def __init__(
        self,
        rise_tolerance: float = 0.10,
        max_damp_per_step: float = 0.15,
        scale_floor: float = 0.5,
        allow_recovery: bool = False,
        recovery_ceiling: float = 0.85,
        recover_tolerance: float = 0.03,
        window_size: int = 100,
        warmup_steps: int = 5000,
        param_str_match: Optional[str] = None,
        allow_missing_history: bool = False,
        scale_bounds_mode: str = SCALE_BOUNDS_DYNAMIC,
        rewrite_checkpoint_stats: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert rise_tolerance >= 0.0, f"rise_tolerance must be >= 0; got {rise_tolerance}"
        assert 0.0 < max_damp_per_step <= 1.0, (
            f"max_damp_per_step must be in (0, 1]; got {max_damp_per_step}"
        )
        assert 0.0 < scale_floor <= 1.0, (
            f"scale_floor must be in (0, 1]; got {scale_floor}"
        )
        assert scale_floor <= recovery_ceiling <= 1.0, (
            f"recovery_ceiling must be in [scale_floor, 1]; got {recovery_ceiling}"
        )
        assert not allow_recovery or 0.0 <= recover_tolerance < rise_tolerance, (
            f"recover_tolerance must be in [0, rise_tolerance); got {recover_tolerance}"
        )
        assert warmup_steps > window_size, (
            f"warmup_steps ({warmup_steps}) must be greater than window_size ({window_size})"
        )
        assert scale_bounds_mode in (self.SCALE_BOUNDS_DYNAMIC, self.SCALE_BOUNDS_EFFECTIVE), (
            "scale_bounds_mode must be either "
            f"{self.SCALE_BOUNDS_DYNAMIC!r} or {self.SCALE_BOUNDS_EFFECTIVE!r}; "
            f"got {scale_bounds_mode!r}"
        )
        self.rise_tolerance = rise_tolerance
        self.max_damp_per_step = max_damp_per_step
        self.scale_floor = scale_floor
        self.allow_recovery = allow_recovery
        self.recovery_ceiling = recovery_ceiling
        self.recover_tolerance = recover_tolerance
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.param_str_match = param_str_match
        self.allow_missing_history = allow_missing_history
        self.scale_bounds_mode = scale_bounds_mode
        self._rewrite_checkpoint_stats = self._normalize_rewrite_checkpoint_stats(
            rewrite_checkpoint_stats
        )

        # The UpdateBias algorithm we drive; resolved at fit_start.
        self._update_bias = None
        # MoE name tag (e.g. "layer_29") -> real decoder-layer index.
        self._tag_to_block: Dict[str, int] = {}
        # real decoder-layer index -> its trainable parameter names (for param_str_match).
        self._block_param_names: Dict[int, List[str]] = {}
        # blocks we can actually control, in a stable order for collective ops.
        self._controllable_blocks: List[int] = []
        # block -> rolling window of recent raw max-violation samples (seeding /
        # checkpointing / warming the KF before the first action).
        self._max_vio_history: Dict[int, deque] = {}
        # block -> robust local-trend Kalman filter (the true-signal estimator).
        self._kf: Dict[int, _LocalTrendKF] = {}
        # block -> DYNAMIC bias-rate scale in [scale_floor, 1.0] (1.0 = nominal).
        self._block_scale: Dict[int, float] = {}
        # block -> running-min envelope of the KF level since warmup (ratchets down).
        self._env: Dict[int, Optional[float]] = {}
        # block -> STATIC base scale from UpdateBias config; re-read each fit_start.
        self._block_base_scale: Dict[int, float] = {}

        # Checkpoint-restored state, re-applied in _fit_start.
        self._loaded_block_scale: Dict[int, float] = {}
        self._loaded_max_vio_history: Dict[int, List[float]] = {}
        self._loaded_kf_state: Dict[int, dict] = {}
        self._loaded_env: Dict[int, float] = {}
        self._loaded_scale_bounds_mode: Optional[str] = None
        self._rewrite_checkpoint_stats_applied = False

    def _new_kf(self) -> _LocalTrendKF:
        return _LocalTrendKF(
            level_process_ratio=self.KF_LEVEL_RATIO,
            trend_process_ratio=self.KF_TREND_RATIO,
            obs_var_halflife=self.KF_OBS_HALFLIFE,
        )

    @staticmethod
    def _block_key(block: Any) -> int:
        if isinstance(block, str):
            block = block.strip()
        try:
            return int(block)
        except (TypeError, ValueError) as e:
            raise ValueError(f"block ids must be integer decoder-layer indices; got {block!r}") from e

    @classmethod
    def _normalize_float_map(cls, name: str, values: Optional[dict]) -> Dict[int, float]:
        if values is None:
            return {}
        return {cls._block_key(block): float(value) for block, value in dict(values).items()}

    @classmethod
    def _normalize_history_map(cls, values: Optional[dict]) -> Dict[int, List[float]]:
        if values is None:
            return {}
        return {
            cls._block_key(block): [float(v) for v in list(history)]
            for block, history in dict(values).items()
        }

    @classmethod
    def _normalize_state_map(cls, values: Optional[dict]) -> Dict[int, dict]:
        if values is None:
            return {}
        return {
            cls._block_key(block): dict(state)
            for block, state in dict(values).items()
        }

    @classmethod
    def _normalize_rewrite_checkpoint_stats(cls, stats: Optional[dict]) -> Optional[dict]:
        if stats is None:
            return None
        stats = dict(stats)
        allowed = {"block_scale", "max_vio_history", "kf_state", "env"}
        unknown = sorted(set(stats) - allowed)
        if unknown:
            raise ValueError(
                "rewrite_checkpoint_stats only supports keys "
                f"{sorted(allowed)}; got unknown keys {unknown}"
            )
        return {
            "block_scale": cls._normalize_float_map("block_scale", stats.get("block_scale")),
            "max_vio_history": cls._normalize_history_map(stats.get("max_vio_history")),
            "kf_state": cls._normalize_state_map(stats.get("kf_state")),
            "env": cls._normalize_float_map("env", stats.get("env")),
        }

    def _fit_start(self, state: State, logger: Logger):
        # Parent populates self.name_tags / self._latest_per_layer_max_vio here; this
        # runs on both fit_start and after_load (resume).
        super()._fit_start(state, logger)

        self._build_tag_to_block()
        self._build_block_param_names(state)

        # A block is controllable if it emits a max-violation tag AND the UpdateBias
        # algorithm actually gates it (has an MoE router there).
        self._update_bias = self._find_update_bias(state)
        moe_blocks = sorted(set(self._tag_to_block.values()))
        if self._update_bias is None or not hasattr(self._update_bias, "set_bias_rate_scale"):
            log.warning(
                "DynamicBlockBiasRateRescaler is enabled but no UpdateBias algorithm "
                "with a `set_bias_rate_scale` API was found; no per-block bias-rate "
                "control will be applied."
            )
            self._controllable_blocks = []
        else:
            gate_blocks = set(self._update_bias.gate_block_indices(state))
            self._controllable_blocks = [b for b in moe_blocks if b in gate_blocks]
            skipped = [b for b in moe_blocks if b not in gate_blocks]
            if skipped:
                log.warning(
                    "DynamicBlockBiasRateRescaler: MoE blocks with a dispatch signal "
                    f"but no UpdateBias router (not controlled): {skipped}"
                )

        # Optionally restrict to blocks whose parameter names match the regex.
        if self.param_str_match is not None and self._controllable_blocks:
            pat = re.compile(self.param_str_match)
            kept = [
                b for b in self._controllable_blocks
                if any(pat.search(n) for n in self._block_param_names.get(b, []))
            ]
            dropped = [b for b in self._controllable_blocks if b not in kept]
            self._controllable_blocks = kept
            log.info(
                f"DynamicBlockBiasRateRescaler param_str_match={self.param_str_match!r} "
                f"keeps blocks {kept}; excludes {dropped}."
            )

        # Read the static base scale from UpdateBias (re-read every fit_start).
        self._block_base_scale = {
            block: self._update_bias.get_base_bias_rate_scale(block)
            for block in self._controllable_blocks
        }
        if self.scale_bounds_mode == self.SCALE_BOUNDS_EFFECTIVE:
            bad_base = {
                block: base
                for block, base in self._block_base_scale.items()
                if float(base) <= 0.0
            }
            if bad_base:
                raise ValueError(
                    "DynamicBlockBiasRateRescaler effective scale mode requires positive "
                    f"static bias-rate base scales; got {bad_base}"
                )

        # Seed the rolling window + KF (own -> sibling DynLR -> rewrite -> raise/empty).
        self._seed_state(state)

        # Push the (possibly restored/seeded) dynamic scales onto UpdateBias.
        for block in self._controllable_blocks:
            self._update_bias.set_bias_rate_scale(
                block,
                self._dynamic_scale_for_update_bias(block, self._block_scale[block]),
            )

        if self._controllable_blocks:
            log.info(
                f"DynamicBlockBiasRateRescaler controlling {len(self._controllable_blocks)} "
                f"decoder layers: {self._controllable_blocks}"
            )

    def _default_block_scale(self, block: int) -> float:
        if self.scale_bounds_mode == self.SCALE_BOUNDS_EFFECTIVE:
            return float(self._block_base_scale.get(block, 1.0))
        return 1.0

    def _seed_block_scale(self, block: int, seed_scale: Dict[int, float], source: str) -> float:
        if block not in seed_scale:
            return self._default_block_scale(block)

        scale = float(seed_scale[block])
        if (
            self._rewrite_checkpoint_stats is not None
            and block in self._rewrite_checkpoint_stats["block_scale"]
        ):
            return scale
        loaded_mode = self._loaded_scale_bounds_mode or self.SCALE_BOUNDS_DYNAMIC
        if not source.startswith("own checkpoint") or loaded_mode == self.scale_bounds_mode:
            return scale

        base = float(self._block_base_scale.get(block, 1.0))
        if loaded_mode == self.SCALE_BOUNDS_DYNAMIC and self.scale_bounds_mode == self.SCALE_BOUNDS_EFFECTIVE:
            return base * scale
        if loaded_mode == self.SCALE_BOUNDS_EFFECTIVE and self.scale_bounds_mode == self.SCALE_BOUNDS_DYNAMIC:
            if base <= 0.0:
                raise ValueError(
                    "Cannot convert checkpointed effective scale to dynamic scale "
                    f"for block {block}: base scale is {base}"
                )
            return scale / base
        return scale

    def _dynamic_scale_for_update_bias(self, block: int, controller_scale: float) -> float:
        if self.scale_bounds_mode == self.SCALE_BOUNDS_EFFECTIVE:
            base = float(self._block_base_scale.get(block, 1.0))
            if base <= 0.0:
                raise ValueError(
                    "Cannot apply effective bias-rate scale for block "
                    f"{block}: static base scale is {base}"
                )
            return float(controller_scale) / base
        return float(controller_scale)

    def _effective_scale(self, block: int) -> float:
        if self.scale_bounds_mode == self.SCALE_BOUNDS_EFFECTIVE:
            return self._block_scale[block]
        return self._block_base_scale[block] * self._block_scale[block]

    def _rewrite_has_history(self, blocks: List[int]) -> bool:
        if self._rewrite_checkpoint_stats is None:
            return False
        rewrite = self._rewrite_checkpoint_stats
        return any(rewrite["max_vio_history"].get(b) for b in blocks) or \
            any(rewrite["kf_state"].get(b) for b in blocks)

    def _apply_rewrite_checkpoint_stats(
        self,
        seed_hist: Dict[int, List[float]],
        seed_scale: Dict[int, float],
        seed_kf: Dict[int, dict],
        seed_env: Dict[int, float],
        blocks: List[int],
    ) -> bool:
        if self._rewrite_checkpoint_stats is None:
            return False

        rewrite = self._rewrite_checkpoint_stats
        block_set = set(blocks)
        unknown = sorted(
            set().union(
                rewrite["block_scale"],
                rewrite["max_vio_history"],
                rewrite["kf_state"],
                rewrite["env"],
            ) - block_set
        )
        if unknown:
            log.warning(
                "DynamicBlockBiasRateRescaler rewrite_checkpoint_stats references "
                f"uncontrolled blocks (ignored): {unknown}"
            )

        for block, value in rewrite["max_vio_history"].items():
            if block in block_set:
                seed_hist[block] = value
        for block, value in rewrite["block_scale"].items():
            if block in block_set:
                seed_scale[block] = value
        for block, value in rewrite["kf_state"].items():
            if block in block_set:
                seed_kf[block] = value
        for block, value in rewrite["env"].items():
            if block in block_set:
                seed_env[block] = value

        return True

    def _seed_state(self, state: State):
        """Resolve the rolling window, KF state and dynamic scales per the ladder."""
        blocks = self._controllable_blocks

        have_own = any(self._loaded_max_vio_history.get(b) for b in blocks) or \
            any(self._loaded_kf_state.get(b) for b in blocks)
        if have_own:
            seed_hist = self._loaded_max_vio_history
            seed_scale = self._loaded_block_scale
            seed_kf = self._loaded_kf_state
            seed_env = self._loaded_env
            source = "own checkpoint"
        else:
            sibling_hist = self._sibling_dynlr_history(state)
            if sibling_hist is not None and any(sibling_hist.get(b) for b in blocks):
                seed_hist = sibling_hist
                seed_scale = {}
                seed_kf = {}
                seed_env = {}
                source = "sibling DynamicBlockLRRescaler window"
            elif blocks and self._rewrite_has_history(blocks):
                seed_hist = {}
                seed_scale = {}
                seed_kf = {}
                seed_env = {}
                source = "rewrite_checkpoint_stats"
            elif blocks and state.timestamp.batch.value > 0 and not self.allow_missing_history:
                raise RuntimeError(
                    "DynamicBlockBiasRateRescaler resumed an in-progress run "
                    f"(batch={state.timestamp.batch.value}) with no max-violation "
                    "history: neither its own checkpointed window nor a sibling "
                    "DynamicBlockLRRescaler window is available. Refusing to run blind. "
                    "Enable DynamicBlockLRRescaler (so its window can be reused), inject "
                    "a window, or set `allow_missing_history=True` to start empty."
                )
            else:
                seed_hist = {}
                seed_scale = {}
                seed_kf = {}
                seed_env = {}
                source = "empty (fresh start)"

        rewrite_applied = self._apply_rewrite_checkpoint_stats(
            seed_hist=seed_hist,
            seed_scale=seed_scale,
            seed_kf=seed_kf,
            seed_env=seed_env,
            blocks=blocks,
        )
        self._rewrite_checkpoint_stats_applied = rewrite_applied
        if rewrite_applied:
            source = f"{source} + rewrite_checkpoint_stats"

        self._max_vio_history = {
            block: deque(seed_hist.get(block, ()), maxlen=self.window_size)
            for block in blocks
        }
        self._block_scale = {
            block: self._seed_block_scale(block, seed_scale, source)
            for block in blocks
        }

        # Rebuild the Kalman filters and running-min envelope. Restore checkpointed KF
        # state if present, else replay the seed window through a fresh slow KF so the
        # estimator is warm before the first decision. The envelope is restored from the
        # checkpoint, or (when seeding from a sibling/own window without a saved envelope)
        # reconstructed as the minimum KF level over the replayed window.
        self._kf = {}
        self._env = {}
        for block in blocks:
            kf = self._new_kf()
            env = seed_env.get(block)
            if seed_kf.get(block):
                kf.load_state_dict(seed_kf[block])
            else:
                lo = None
                for z in self._max_vio_history[block]:
                    L, _ = kf.update(float(z))
                    lo = L if lo is None else min(lo, L)
                if env is None and lo is not None:
                    env = lo
            self._kf[block] = kf
            self._env[block] = env

        if blocks:
            log.info(f"DynamicBlockBiasRateRescaler window seeded from: {source}")

    def _find_update_bias(self, state: State):
        for algo in getattr(state, "algorithms", []) or []:
            if UpdateBias is not None and isinstance(algo, UpdateBias):
                return algo
            if type(algo).__name__ == "UpdateBias":
                return algo
        return None

    def _sibling_dynlr_history(self, state: State) -> Optional[Dict[int, List[float]]]:
        """The live sibling DynamicBlockLRRescaler's max-violation window, if any."""
        for cb in getattr(state, "callbacks", []) or []:
            if isinstance(cb, DynamicBlockLRRescaler):
                loaded = getattr(cb, "_loaded_max_vio_history", None)
                if loaded:
                    return {b: list(h) for b, h in loaded.items()}
                live = getattr(cb, "_max_vio_history", None)
                if live:
                    return {b: list(h) for b, h in live.items()}
                return None
        return None

    def _build_tag_to_block(self):
        self._tag_to_block = {}
        for tag in self.name_tags:
            match = _LAYER_TAG_RE.match(tag)
            # Main decoder layers only; MTP layers ("mtp_layer_*") are skipped.
            if match is not None:
                self._tag_to_block[tag] = int(match.group(1))

    def _build_block_param_names(self, state: State):
        """Map each decoder-layer index to its trainable parameter names (for regex)."""
        names: Dict[int, List[str]] = defaultdict(list)
        for name, param in state.model.named_parameters():
            if not param.requires_grad:
                continue
            if "mtp" in name:
                continue
            match = _LAYER_INDEX_RE.search(name)
            if match is not None:
                names[int(match.group(1))].append(name)
        self._block_param_names = dict(names)

    def batch_end(self, state: State, logger: Logger):
        super().batch_end(state, logger)
        # The parent only refreshes its statistics on its logging interval.
        if state.timestamp.batch.value % self.batch_log_interval != 0:
            return

        self._capture_max_vios()

        if state.timestamp.batch.value >= self.warmup_steps and self._controllable_blocks:
            self._update_block_rates(state, logger)

        if self._controllable_blocks:
            self._log_block_rates(state, logger)

    def _capture_max_vios(self):
        """Append the latest raw sample to each window and advance its Kalman filter."""
        for tag, value in self._latest_per_layer_max_vio.items():
            block = self._tag_to_block.get(tag)
            if block in self._max_vio_history:
                v = float(value)
                self._max_vio_history[block].append(v)
                self._kf[block].update(v)

    def _update_block_rates(self, state: State, logger: Logger):
        """Damp a block when its KF level rises above its running-min envelope.

        Ratchet the envelope down to the new low; if the level sits more than
        ``rise_tolerance`` above it, cut the gain (proportional, capped at
        ``max_damp_per_step``). By default (no recovery) the gain only ever decreases or
        holds -> non-increasing and non-oscillating. All quantities are ratios, so the
        rule is identical at any baseline max-violation / model size.
        """
        eps = 1e-6

        # Per block: fractional excess above the envelope (col 0) and relative slope
        # (col 1), packed for one MAX all-reduce. KFs get identical (all-reduced) inputs so
        # agree across ranks; MAX guards float drift and is conservative (most damping).
        stat = torch.zeros(
            len(self._controllable_blocks), 2, dtype=torch.float32,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        eligible = [False] * len(self._controllable_blocks)
        for i, block in enumerate(self._controllable_blocks):
            kf = self._kf[block]
            if kf.L is None or kf.n < min(self.window_size, 20):  # need a warm KF
                continue
            env = self._env.get(block)
            env = kf.L if env is None else min(env, kf.L)  # ratchet reference down
            self._env[block] = env
            stat[i, 0] = (kf.L - env) / max(env, eps)
            stat[i, 1] = kf.v / max(kf.L, eps)
            eligible[i] = True

        if dist.get_world_size() > 1:
            dist.all_reduce(stat, reduce_operation="MAX")

        changed: List[int] = []
        for i, block in enumerate(self._controllable_blocks):
            if not eligible[i]:
                continue
            scale = self._block_scale[block]
            excess = float(stat[i, 0].item())
            rel_slope = float(stat[i, 1].item())
            if excess > self.rise_tolerance:  # rose too far -> damp, proportional to overshoot
                cut = min(self.max_damp_per_step,
                          self._DAMP_GAIN * (excess - self.rise_tolerance))
                # If a manually seeded scale already sits below the floor, keep the
                # one-way ratchet invariant and do not raise it during a damp step.
                new_scale = scale if scale <= self.scale_floor else max(
                    self.scale_floor, scale * (1.0 - cut)
                )
            elif (self.allow_recovery and excess < self.recover_tolerance
                    and rel_slope <= 0.0 and scale < self.recovery_ceiling):
                # opt-in: reclaim gain only while hugging the envelope and not rising
                new_scale = min(self.recovery_ceiling, scale * (1.0 + self._RECOVERY_STEP))
            else:
                new_scale = scale
            if new_scale != scale:
                self._block_scale[block] = new_scale
                self._update_bias.set_bias_rate_scale(
                    block,
                    self._dynamic_scale_for_update_bias(block, new_scale),
                )
                if new_scale < scale:
                    changed.append(block)

        if changed:
            log.info(
                "DynamicBlockBiasRateRescaler damped bias rate for blocks "
                f"{changed} (dynamic scales: "
                f"{ {b: round(self._block_scale[b], 4) for b in changed} })"
            )

    def _log_block_rates(self, state: State, logger: Logger):
        """Log each block's effective scale/rate plus the KF estimates.

        ``dynamic_block_bias_rate/scale/block_{i}`` is the EFFECTIVE multiplier
        ``base * dynamic``; ``.../rate/block_{i}`` the effective rate
        ``global_rate * base * dynamic``. The estimator/controller signals are also logged
        per block: ``.../est_level`` (KF level), ``.../env`` (running-min envelope) and
        ``.../excess`` (the scale-free rise above the envelope the controller acts on).
        """
        eps = 1e-6
        global_rate = float(getattr(self._update_bias, "rate", 0.0) or 0.0)
        metrics = {}
        for block in self._controllable_blocks:
            effective_scale = self._effective_scale(block)
            dynamic_scale = self._dynamic_scale_for_update_bias(block, self._block_scale[block])
            metrics[f"dynamic_block_bias_rate/scale/block_{block}"] = effective_scale
            metrics[f"dynamic_block_bias_rate/rate/block_{block}"] = global_rate * effective_scale
            metrics[f"dynamic_block_bias_rate/controller_scale/block_{block}"] = self._block_scale[block]
            metrics[f"dynamic_block_bias_rate/dynamic_scale/block_{block}"] = dynamic_scale
            kf = self._kf.get(block)
            env = self._env.get(block)
            if kf is not None and kf.L is not None:
                metrics[f"dynamic_block_bias_rate/est_level/block_{block}"] = kf.L
                if env is not None:
                    metrics[f"dynamic_block_bias_rate/env/block_{block}"] = env
                    metrics[f"dynamic_block_bias_rate/excess/block_{block}"] = (
                        (kf.L - env) / max(env, eps)
                    )
        logger.log_metrics(metrics)

    def state_dict(self) -> dict:
        return {
            "scale_bounds_mode": self.scale_bounds_mode,
            "block_scale": dict(self._block_scale),
            "max_vio_history": {
                block: list(history)
                for block, history in self._max_vio_history.items()
            },
            "kf_state": {block: kf.state_dict() for block, kf in self._kf.items()},
            "env": {b: e for b, e in self._env.items() if e is not None},
            "base_scale_at_save": dict(self._block_base_scale),
            "rewrite_checkpoint_stats_applied": self._rewrite_checkpoint_stats_applied,
        }

    def load_state_dict(self, state: dict) -> None:
        self._loaded_scale_bounds_mode = state.get("scale_bounds_mode")
        self._loaded_block_scale = self._normalize_float_map(
            "block_scale", state.get("block_scale", {})
        )
        self._loaded_max_vio_history = {
            block: list(history)
            for block, history in self._normalize_history_map(
                state.get("max_vio_history", {})
            ).items()
        }
        self._loaded_kf_state = self._normalize_state_map(state.get("kf_state", {}))
        self._loaded_env = self._normalize_float_map("env", state.get("env", {}))
