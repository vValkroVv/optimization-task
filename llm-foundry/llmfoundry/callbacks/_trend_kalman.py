# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free robust local-linear-trend Kalman filter.

Kept in its own module (no composer/torch imports) so it can be unit-tested and
simulated standalone, and reused by :mod:`dynamic_block_bias_rate_rescaler`.
"""

from typing import Optional, Tuple

__all__ = ["_LocalTrendKF"]


class _LocalTrendKF:
    """Robust local-linear-trend Kalman filter over a scalar stream.

    State ``x = [level, slope]`` with a constant-velocity model::

        level_{t+1} = level_t + slope_t + w_l
        slope_{t+1} = slope_t          + w_s
        z_t         = level_t          + v_t   (noisy observation)

    Returns a smoothed ``level`` and ``slope`` (per-sample trend) recovered from the
    noisy, spike-prone observation stream.

    * **Adaptive observation noise.** ``R`` is estimated online from the innovation
      sequence (EWMA of ``innovation^2 - predicted_var``), so the caller need not
      hand-tune a noise level that depends on the signal scale. Process noise is set as
      dimensionless *ratios* of ``R`` (``level_process_ratio`` / ``trend_process_ratio``),
      making the filter scale-free.
    * **Outlier rejection.** An innovation beyond ``outlier_gate`` sigma has its
      observation variance inflated for that step, so a single spike barely moves the
      state; the adaptive-``R`` update also skips such steps.

    All state is plain Python floats/lists so it checkpoints trivially.
    """

    def __init__(
        self,
        level_process_ratio: float = 0.1,
        trend_process_ratio: float = 0.01,
        outlier_gate: Optional[float] = 3.0,
        obs_var_halflife: float = 20.0,
    ):
        self.rl = float(level_process_ratio)
        self.rt = float(trend_process_ratio)
        self.gate = outlier_gate
        self.obs_var_retain = 0.5 ** (1.0 / max(1e-6, obs_var_halflife))
        self.reset()

    def reset(self):
        self.L: Optional[float] = None  # estimated true level
        self.v: float = 0.0             # estimated slope (per sample)
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        self.R: Optional[float] = None  # observation-noise variance (adaptive)
        self.n: int = 0

    def update(self, z: float) -> Tuple[float, float]:
        z = float(z)
        if self.L is None:
            self.L, self.v, self.n = z, 0.0, 1
            self.R = max((0.1 * abs(z) + 1e-6) ** 2, 1e-12)
            return self.L, self.v

        # --- predict: x <- F x,  P <- F P F^T + Q  (F = [[1,1],[0,1]]) ---
        L_pred = self.L + self.v
        v_pred = self.v
        P = self.P
        p00 = P[0][0] + P[1][0] + P[0][1] + P[1][1]
        p01 = P[0][1] + P[1][1]
        p10 = P[1][0] + P[1][1]
        p11 = P[1][1]
        R = self.R if self.R is not None else max((0.1 * abs(self.L) + 1e-6) ** 2, 1e-12)
        p00 += (self.rl ** 2) * R  # level process noise
        p11 += (self.rt ** 2) * R  # trend process noise

        # --- innovation + robust gating (H = [1, 0]) ---
        y = z - L_pred
        S = p00 + R
        maha = (y * y) / max(S, 1e-12)
        inlier = self.gate is None or maha <= self.gate ** 2
        if not inlier:
            R_eff = R * (maha / (self.gate ** 2))
            S = p00 + R_eff

        # --- update ---
        k0 = p00 / S
        k1 = p10 / S
        self.L = L_pred + k0 * y
        self.v = v_pred + k1 * y
        n00 = (1.0 - k0) * p00
        n01 = (1.0 - k0) * p01
        n10 = p10 - k1 * p00
        n11 = p11 - k1 * p01
        self.P = [[n00, n01], [n10, n11]]
        self.n += 1

        # --- adapt observation noise from inlier innovations only ---
        if inlier:
            resid_var = max(y * y - p00, 1e-12)
            self.R = self.obs_var_retain * R + (1.0 - self.obs_var_retain) * resid_var
        return self.L, self.v

    def state_dict(self) -> dict:
        return {"L": self.L, "v": self.v, "P": self.P, "R": self.R, "n": self.n}

    def load_state_dict(self, s: dict):
        self.L = s.get("L")
        self.v = float(s.get("v", 0.0))
        self.P = [list(r) for r in s.get("P", [[1.0, 0.0], [0.0, 1.0]])]
        self.R = s.get("R")
        self.n = int(s.get("n", 0))
