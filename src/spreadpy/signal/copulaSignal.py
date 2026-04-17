from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import kendalltau


from spreadpy.signal.signal import SignalGenerator, Signal, Direction
from spreadpy.spread.spreadSeries import SpreadSeries


class CopulaSignal(SignalGenerator):
    """
    Copula-based entry / exit signal generator.

    Uses the conditional CDF of a fitted bivariate copula to detect
statistical mispricings between the two legs.

    In-sample fitting:
        1. Convert prices to pseudo-observations u_t = F̂_y(y_t),
           v_t = F̂_x(x_t) via empirical ranks (Hazen formula).
        2. Estimate Kendall's τ and invert to copula parameter θ:
               Gaussian:  θ = sin(π/2 · τ)
               Clayton:   θ = 2τ / (1 - τ)   (θ > 0)
               Gumbel:    θ = 1 / (1 - τ)    (θ ≥ 1)

    Entry logic (out-of-sample):
        LONG  if C(u_t | v_t) < entry_prob         (y cheap relative to x)
        SHORT if C(u_t | v_t) > 1 - entry_prob     (y expensive relative to x)

    Exit uses the rolling z-score: FLAT when |z| < exit_zscore or |z| > stop_zscore.
    The z-score is also passed to the :class:`LinearSizer` for sizing.

    :param str family: Copula family — ``'gaussian'``, ``'clayton'``, or ``'gumbel'``.
    :param int window: Rolling window for the z-score computation (position sizing only).
    :param float entry_prob: Tail probability threshold for entry (default 0.10).
        Fires when the conditional CDF is below ``entry_prob`` or above ``1 - entry_prob``.
    :param float exit_zscore: Exit position when |z| drops below this value.
    :param float stop_zscore: Stop-loss when |z| exceeds this value.
    """

    SUPPORTED_FAMILIES = {"gaussian", "clayton", "gumbel"}

    def __init__(
        self,
        family: str = "gaussian",
        window: int = 60,
        entry_prob: float = 0.10,
        exit_zscore: float = 0.5,
        stop_zscore: float = 4.0,
    ) -> None:
        if family not in self.SUPPORTED_FAMILIES:
            raise ValueError(f"family must be one of {self.SUPPORTED_FAMILIES}")
        self.family = family
        self.window = window
        self.entry_prob = entry_prob
        self.exit_zscore = exit_zscore
        self.stop_zscore = stop_zscore

        # Calibrated attributes
        self._theta: Optional[float] = None
        self._kendall_tau: Optional[float] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, spread: SpreadSeries) -> "CopulaSignal":
        """
        Calibrate copula parameter θ from in-sample data using
        Kendall's τ inversion method.
        """
        y_vals = spread.y.values.astype(float)
        x_vals = spread.x.values.astype(float)

        tau, _ = kendalltau(y_vals, x_vals)
        self._kendall_tau = float(tau)
        self._theta = self._tau_to_theta(tau)
        return self

    def _tau_to_theta(self, tau: float) -> float:
        """Convert Kendall's τ to copula parameter θ."""
        if self.family == "gaussian":
            # θ = sin(π/2 * τ)
            return float(np.sin(np.pi / 2 * tau))
        elif self.family == "clayton":
            # θ = 2τ / (1 - τ),  θ > 0
            if tau <= 0:
                return 0.01
            return float(2 * tau / (1 - tau))
        elif self.family == "gumbel":
            # θ = 1 / (1 - τ),  θ >= 1
            if tau >= 1:
                return 100.0
            return float(1.0 / (1 - tau))
        raise ValueError(f"Unknown family: {self.family}")

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, spread: SpreadSeries) -> pd.Series:
        if self._theta is None:
            raise RuntimeError("Call fit() before generate()")

        y_vals = spread.y.series.values.astype(float)
        x_vals = spread.x.series.values.astype(float)
        residuals = spread.residuals

        # Rolling z-score for sizing
        roll_mu = residuals.rolling(self.window).mean()
        roll_sigma = residuals.rolling(self.window).std().replace(0, np.nan)
        zscore = (residuals - roll_mu) / roll_sigma

        # Pseudo-observations (marginal CDFs via empirical rank)
        n = len(y_vals)
        u_all = self._empirical_cdf(y_vals)
        v_all = self._empirical_cdf(x_vals)

        signals = []
        prev_direction = Direction.FLAT

        for i, ts in enumerate(spread.index):
            z = float(zscore.iloc[i]) if i < len(zscore) else float("nan")

            if np.isnan(z) or i < self.window:
                signals.append(Signal(Direction.FLAT, float("nan"), ts))
                continue

            u_t = float(u_all[i])
            v_t = float(v_all[i])
            cond_u_given_v = self._conditional_cdf(u_t, v_t)
            cond_v_given_u = self._conditional_cdf(v_t, u_t)

            # Exit / stop
            if prev_direction != Direction.FLAT:
                z_abs = abs(z)
                if z_abs < self.exit_zscore or z_abs > self.stop_zscore:
                    sig = Signal(Direction.FLAT, z, ts,
                                 prob=cond_u_given_v, is_entry=False)
                    prev_direction = Direction.FLAT
                    signals.append(sig)
                    continue

            # Entry: extreme conditional probability signals mispricing
            if cond_u_given_v < self.entry_prob:
                # y likely too low relative to x → long spread
                direction = Direction.LONG
                is_entry = prev_direction != Direction.LONG
            elif cond_u_given_v > (1 - self.entry_prob):
                direction = Direction.SHORT
                is_entry = prev_direction != Direction.SHORT
            else:
                direction = prev_direction
                is_entry = False

            prob = cond_u_given_v
            signals.append(Signal(direction, z, ts, prob=prob, is_entry=is_entry))
            prev_direction = direction

        return pd.Series(signals, index=spread.index, name="signal")

    # ------------------------------------------------------------------
    # Copula helpers
    # ------------------------------------------------------------------

    def _empirical_cdf(self, x: np.ndarray) -> np.ndarray:
        """Empirical CDF via fractional ranks, Hazen formula."""
        n = len(x)
        ranks = stats.rankdata(x)
        return (ranks - 0.5) / n

    def _conditional_cdf(self, u: float, v: float) -> float:
        """
        C(u | v) = ∂C(u,v)/∂v  (h-function of the copula).
        Computed numerically for Clayton and Gumbel;
        analytically for Gaussian.
        """
        eps = 1e-9
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        theta = self._theta

        if self.family == "gaussian":
            # Conditional: Φ( (Φ⁻¹(u) - θ·Φ⁻¹(v)) / √(1-θ²) )
            rho = np.clip(theta, -0.999, 0.999)
            phi_u = stats.norm.ppf(u)
            phi_v = stats.norm.ppf(v)
            denom = np.sqrt(max(1 - rho ** 2, 1e-12))
            return float(stats.norm.cdf((phi_u - rho * phi_v) / denom))

        elif self.family == "clayton":
            # h-function: v^(-θ-1) * (u^-θ + v^-θ - 1)^(-1-1/θ)
            theta = max(theta, 1e-6)
            inner = max(u ** (-theta) + v ** (-theta) - 1, eps)
            h = (v ** (-theta - 1)) * (inner ** (-1 - 1 / theta))
            return float(np.clip(h, 0, 1))

        elif self.family == "gumbel":
            # Numerical derivative of C(u,v) w.r.t. v
            dv = 1e-5
            c_plus  = self._gumbel_cdf(u, min(v + dv, 1 - eps))
            c_minus = self._gumbel_cdf(u, max(v - dv, eps))
            return float(np.clip((c_plus - c_minus) / (2 * dv), 0, 1))

        raise ValueError(f"Unknown family: {self.family}")

    def _gumbel_cdf(self, u: float, v: float) -> float:
        """Bivariate Gumbel copula CDF."""
        eps = 1e-9
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        theta = max(self._theta, 1.0)
        lu = (-np.log(u)) ** theta
        lv = (-np.log(v)) ** theta
        return float(np.exp(-((lu + lv) ** (1 / theta))))
