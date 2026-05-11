from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from spreadpy.signal.signal import Direction, Signal
from spreadpy.sizing.positionSizer import PositionSizer
from spreadpy.spread.spreadSeries import SpreadSeries


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


class InverseVolSizer(PositionSizer):
    """
    Markowitz / inverse-volatility position sizer.

    Sizes positions so that a 1-σ adverse spread move costs ``target_vol``
    of current capital:

        frac_t = min(target_vol / σ_t, f_max)

    where σ_t is the rolling standard deviation of spread residuals over
    the last ``window`` bars (no lookahead), consistent with the rolling
    z-score used by :class:`ZScoreSignal`.

    :meth:`fit` must be called before :meth:`size`. The engine passes the
    combined train + eval spread so that σ_t is already warmed up at the
    start of the evaluation period.

    Parameters
    ----------
    window : int
        Rolling window for spread vol estimation (bars). Default 60.
    target_vol : float
        Target capital fraction at risk for a 1-σ spread move. Default 0.02.
    f_max : float
        Hard cap on the capital fraction. Default 0.5.
    """

    def __init__(
        self,
        window: int = 60,
        target_vol: float = 0.02,
        f_max: float = 0.5,
    ) -> None:
        self.window = window
        self.target_vol = target_vol
        self.f_max = f_max
        self._sigma_ts: Optional[pd.Series] = None

    def fit(self, spread: SpreadSeries) -> "InverseVolSizer":
        """
        Precompute the rolling vol series from the spread residuals.

        Call with the combined train + eval spread so the rolling window
        is warmed up before the first eval bar.
        """
        residuals = spread.residuals
        self._sigma_ts = residuals.rolling(self.window).std()
        return self

    def size(
        self,
        signal: Signal,
        price_y: float,
        price_x: float,
        hedge_ratio: float,
        capital: float = 0.0,
    ) -> Tuple[float, float]:
        if self._sigma_ts is None:
            raise RuntimeError("InverseVolSizer.fit() must be called before size().")

        if signal.direction == Direction.FLAT or np.isnan(signal.zscore):
            return 0.0, 0.0
        if capital <= 0.0:
            return 0.0, 0.0

        sigma = self._sigma_ts.get(signal.timestamp, np.nan)
        if np.isnan(sigma) or sigma <= 0.0:
            return 0.0, 0.0

        frac = min(self.target_vol / sigma, self.f_max)
        return _quantities(frac, 1, price_y, price_x, hedge_ratio)
