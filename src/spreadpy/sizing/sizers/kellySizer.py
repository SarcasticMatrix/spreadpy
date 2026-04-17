from typing import Callable, Optional, Tuple

import numpy as np
from scipy.stats import norm

from spreadpy.signal.signal import Signal, Direction
from spreadpy.sizing import PositionSizer



class KellySizer(PositionSizer):
    """
    Position sizer that allocates a Kelly-optimal fraction of current capital.

    At each entry the notional allocated to the y-leg is:

        notional_y = W_t · f*

    where W_t is the current portfolio equity and f* = kelly_fraction(z0, z_revert, f_max).

    The x-leg quantity mirrors the LinearSizer convention:
        qty_y = notional_y / price_y
        qty_x = notional_y · |β| / price_x

    :param float z0: Entry threshold used in the Kelly derivation.
    :param float z_revert: Reversion target used in the Kelly derivation.
    :param float f_max: Hard cap on the Kelly fraction (default 0.25 = 25% of capital).
    """

    def __init__(
        self,
        z0: float,
        z_revert: float = 0.0,
        f_max: float = 0.5,
    ) -> None:
        self.z0       = z0
        self.z_revert = z_revert
        self.f_max    = f_max
        self._frac    = kelly_fraction(z0, z_revert, f_max)   # pre-compute (fixed params)

    def size(
        self,
        signal: Signal,
        price_y: float,
        price_x: float,
        hedge_ratio: float,
        capital: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Compute absolute quantities for each leg using the Kelly fraction.

        Returns (0, 0) when the signal is flat, the Kelly fraction is 0,
        or capital is non-positive.
        """
        if signal.direction == Direction.FLAT or np.isnan(signal.zscore):
            return 0.0, 0.0
        if capital <= 0.0 or self._frac <= 0.0:
            return 0.0, 0.0

        notional_y = capital * self._frac
        qty_y = notional_y / price_y if price_y > 0 else 0.0
        qty_x = (notional_y * abs(hedge_ratio)) / price_x if price_x > 0 else 0.0
        return qty_y, qty_x


def kelly_fraction(
    z0: float,
    z_revert: float,
    f_max: float = 0.5,
) -> float:
    """
    Kelly-optimal fraction f* of capital to allocate for a mean-reversion trade.

    Derived via a second-order Taylor expansion of E[log(1 + f·G)] where
    G = z_entry - z_revert is the P&L per unit of capital, and z_entry is drawn
    from the truncated normal z_entry | z_entry > z0.

    Using truncated-normal moments:
        λ(z0)  = φ(z0) / (1 - Φ(z0))          (inverse Mills ratio)
        E[G]   = z0 + λ(z0) - z_revert
        E[G²]  = 1 - λ(z0)·(λ(z0) - z0) + E[G]²

    Closed-form Kelly fraction:
        f* = E[G] / E[G²],  capped at f_max

    If E[G] ≤ 0 the signal is not favourable and the function returns 0.

    :param float z0: Entry threshold (e.g. 1.5).
    :param float z_revert: Target reversion level (e.g. 0.0).
    :param float f_max: Hard cap on the returned fraction (default 0.25).
    :returns: Kelly fraction f* ∈ [0, f_max].
    :rtype: float
    """
    lam   = norm.pdf(z0) / (1.0 - norm.cdf(z0))   # inverse Mills ratio
    mu_g  = z0 + lam - z_revert
    if mu_g <= 0.0:
        return 0.0
    e_g2  = 1.0 - lam * (lam - z0) + mu_g ** 2
    if e_g2 <= 0.0:
        return 0.0
    return float(min(mu_g / e_g2, f_max))