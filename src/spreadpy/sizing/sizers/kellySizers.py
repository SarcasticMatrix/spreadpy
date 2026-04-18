"""
Kelly-based position sizers for spread mean-reversion trading.

Both classes derive f* from the second-order Kelly criterion:

    f* = argmax_f E[log(1 + f·G)] ≈ E[G] / E[G²]

where the approximation follows from log(1+x) ≈ x - x²/2, giving:

    E[G] / E[G²] = E[G] / (Var(G) + E[G]²)

The spread z-score is modelled as z_t = (X_t - μ_t) / σ_t ~ N(0,1).
We trade mean-reversion: short when z_t ≥ z_entry, target z_revert < z_entry.

Inverse Mills ratios
--------------------
Left truncation at a (z ≥ a):
    λ₊(a) = φ(a) / (1 − Φ(a))

Right truncation at b (z ≤ b):
    λ₋(b) = φ(b) / Φ(b)

where φ and Φ are the standard normal PDF and CDF.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm

from spreadpy.signal.signal import Direction, Signal
from spreadpy.sizing.positionSizer import PositionSizer


# ---------------------------------------------------------------------------
# Utility: inverse Mills ratios
# ---------------------------------------------------------------------------

def _mills_left(a: float) -> float:
    """λ₊(a) = φ(a) / (1 − Φ(a))  — left-truncation at a."""
    return float(norm.pdf(a) / (1.0 - norm.cdf(a)))


def _mills_right(b: float) -> float:
    """λ₋(b) = φ(b) / Φ(b)  — right-truncation at b."""
    return float(norm.pdf(b) / norm.cdf(b))


# ---------------------------------------------------------------------------
# Shared sizing logic
# ---------------------------------------------------------------------------

def _quantities(
    frac: float,
    capital: float,
    price_y: float,
    price_x: float,
    hedge_ratio: float,
) -> Tuple[float, float]:
    notional_y = capital * frac
    qty_y = notional_y / price_y if price_y > 0.0 else 0.0
    qty_x = (notional_y * abs(hedge_ratio)) / price_x if price_x > 0.0 else 0.0
    return qty_y, qty_x


# ---------------------------------------------------------------------------
# Method 1 — KellyTruncatedEntry
# ---------------------------------------------------------------------------

class KellyTruncatedEntry(PositionSizer):
    """
    Kelly sizer where the **entry level** z is the only random quantity.

    Model
    -----
    z ~ N(0,1) truncated to [z_entry, +∞), with density:

        p(z) = φ(z) / (1 − Φ(z_entry))  ·  1_{z ≥ z_entry}

    The reversion target z_revert is treated as deterministic.
    Gain: G = z − z_revert.

    Moments  (λ₊ = λ₊(z_entry))
    -----------------------------
        E[G]   = λ₊ − z_revert
        Var(z) = 1 − λ₊(λ₊ − z_entry)
        E[G²]  = Var(z) + E[G]²

    Kelly fraction
    --------------
        f* = (λ₊ − z_revert) / (1 − λ₊(λ₊ − z_entry) + (λ₊ − z_revert)²)

    This fraction is constant (does not depend on the observed z_t at entry).

    Parameters
    ----------
    z_entry : float
        Entry threshold; we go short when z_t ≥ z_entry.
    z_revert : float
        Deterministic reversion target (default 0.0).
    f_max : float
        Hard cap on the Kelly fraction (default 0.5).
    """

    def __init__(
        self,
        z_entry: float,
        z_revert: float = 0.0,
        f_max: float = 0.5,
    ) -> None:
        self.z_entry  = z_entry
        self.z_revert = z_revert
        self.f_max    = f_max
        self._frac    = self._kelly()

    def _kelly(self) -> float:
        lam_plus = _mills_left(self.z_entry)
        mu_g     = lam_plus - self.z_revert
        if mu_g <= 0.0:
            return 0.0
        var_z = 1.0 - lam_plus * (lam_plus - self.z_entry)
        e_g2  = var_z + mu_g ** 2
        if e_g2 <= 0.0:
            return 0.0
        return float(min(mu_g / e_g2, self.f_max))

    def size(
        self,
        signal: Signal,
        price_y: float,
        price_x: float,
        hedge_ratio: float,
        capital: float = 0.0,
    ) -> Tuple[float, float]:
        if signal.direction == Direction.FLAT or np.isnan(signal.zscore):
            return 0.0, 0.0
        if capital <= 0.0 or self._frac <= 0.0:
            return 0.0, 0.0
        return _quantities(self._frac, capital, price_y, price_x, hedge_ratio)


# ---------------------------------------------------------------------------
# Method 2 — KellyTruncatedExit
# ---------------------------------------------------------------------------

class KellyTruncatedExit(PositionSizer):
    """
    Kelly sizer where the **reversion level** z̃ is the random quantity.

    Model
    -----
    The effective exit level z̃ ~ N(0,1) truncated to (−∞, z_revert], with density:

        p(z̃) = φ(z̃) / Φ(z_revert)  ·  1_{z̃ ≤ z_revert}

    The observed entry z_t is treated as deterministic.
    Gain: G = z_t − z̃.

    Moments  (λ₋ = λ₋(z_revert))
    -----------------------------
        E[z̃]     = −λ₋
        E[G]      = z_t + λ₋
        Var(z̃)   = 1 − λ₋(λ₋ + z_revert)
        E[G²]     = Var(z̃) + E[G]²

    Kelly fraction
    --------------
        f*(z_t) = (z_t + λ₋) / (1 − λ₋(λ₋ + z_revert) + (z_t + λ₋)²)

    By symmetry of N(0,1), long entries (signal.zscore ≤ −z_entry) use
    z_t = |signal.zscore|: the gain z̃ − z_t for a long trade has identical
    moments to the short gain with the reflected z-score.

    This fraction is recomputed at each date using the observed |z_t|.

    Parameters
    ----------
    z_revert : float
        Right-truncation point for the exit distribution (default 0.0).
    f_max : float
        Hard cap on the Kelly fraction (default 0.5).
    """

    def __init__(
        self,
        z_revert: float = 0.0,
        f_max: float = 0.5,
    ) -> None:
        self.z_revert  = z_revert
        self.f_max     = f_max
        self._lam_minus = _mills_right(z_revert)
        self._var_ztilde = 1.0 - self._lam_minus * (self._lam_minus + z_revert)

    def _kelly(self, z_t: float) -> float:
        mu_g = z_t + self._lam_minus
        if mu_g <= 0.0:
            return 0.0
        e_g2 = self._var_ztilde + mu_g ** 2
        if e_g2 <= 0.0:
            return 0.0
        return float(min(mu_g / e_g2, self.f_max))

    def size(
        self,
        signal: Signal,
        price_y: float,
        price_x: float,
        hedge_ratio: float,
        capital: float = 0.0,
    ) -> Tuple[float, float]:
        if signal.direction == Direction.FLAT or np.isnan(signal.zscore):
            return 0.0, 0.0
        if capital <= 0.0:
            return 0.0, 0.0
        frac = self._kelly(abs(signal.zscore))
        if frac <= 0.0:
            return 0.0, 0.0
        return _quantities(frac, capital, price_y, price_x, hedge_ratio)


# ---------------------------------------------------------------------------
# Method 3 — KellyTruncatedBoth
# ---------------------------------------------------------------------------

class KellyTruncatedBoth(PositionSizer):
    """
    Kelly sizer where **both** the entry z and the reversion z̃ are random and
    independent.

    Model
    -----
        z  ~ N(0,1) truncated to [z_entry, +∞)   — entry level
        z̃ ~ N(0,1) truncated to (−∞, z_revert]   — exit level
        z ⊥ z̃

    Gain: G = z − z̃.

    Moments  (λ₊ = λ₊(z_entry),  λ₋ = λ₋(z_revert))
    ---------------------------------------------------
        E[G]      = λ₊ + λ₋
        Var(z)    = 1 − λ₊(λ₊ − z_entry)
        Var(z̃)   = 1 − λ₋(λ₋ + z_revert)
        Var(G)    = Var(z) + Var(z̃)          (by independence)
        E[G²]     = Var(G) + E[G]²

    Kelly fraction
    --------------
        f* = (λ₊ + λ₋) / (2 − λ₊(λ₊ − z_entry) − λ₋(λ₋ + z_revert) + (λ₊ + λ₋)²)

    This fraction is constant (does not depend on the observed z_t at entry).

    Parameters
    ----------
    z_entry : float
        Entry threshold; we go short when z_t ≥ z_entry.
    z_revert : float
        Right-truncation point for the exit distribution (default 0.0).
    f_max : float
        Hard cap on the Kelly fraction (default 0.5).
    """

    def __init__(
        self,
        z_entry: float,
        z_revert: float = 0.0,
        f_max: float = 0.5,
    ) -> None:
        self.z_entry  = z_entry
        self.z_revert = z_revert
        self.f_max    = f_max
        self._frac    = self._kelly()

    def _kelly(self) -> float:
        lam_plus  = _mills_left(self.z_entry)
        lam_minus = _mills_right(self.z_revert)
        mu_g      = lam_plus + lam_minus
        var_z     = 1.0 - lam_plus  * (lam_plus  - self.z_entry)
        var_ztilde = 1.0 - lam_minus * (lam_minus + self.z_revert)
        e_g2      = var_z + var_ztilde + mu_g ** 2
        if e_g2 <= 0.0:
            return 0.0
        return float(min(mu_g / e_g2, self.f_max))

    def size(
        self,
        signal: Signal,
        price_y: float,
        price_x: float,
        hedge_ratio: float,
        capital: float = 0.0,
    ) -> Tuple[float, float]:
        if signal.direction == Direction.FLAT or np.isnan(signal.zscore):
            return 0.0, 0.0
        if capital <= 0.0 or self._frac <= 0.0:
            return 0.0, 0.0
        return _quantities(self._frac, capital, price_y, price_x, hedge_ratio)
