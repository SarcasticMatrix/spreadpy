"""
spread.py — Layer 2: Spread modelling
HedgeRatioEstimator (abstract), ConstantOLS, RollingOLS,
KalmanFilter, KalmanFilterWithVelocity, SpreadSeries
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from spreadpy.data import PriceTimeSeries
from spreadpy.spread import SpreadSeries


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class HedgeRatioEstimator(ABC):
    """
    Abstract estimator: given two aligned price series (y, x),
    produce a time-series of hedge ratios β such that
        spread_t = y_t - β_t * x_t
    is (ideally) stationary.
    """

    @abstractmethod
    def fit(self, y: PriceTimeSeries, x: PriceTimeSeries) -> pd.Series:
        """
        Fit the model on (y, x) and return a pd.Series of hedge ratios
        aligned with y.index.
        """

    def compute_spread(self, y: PriceTimeSeries, x: PriceTimeSeries) -> "SpreadSeries":
        """Convenience: fit + build SpreadSeries in one call."""
        y_al, x_al = y.align(x)
        beta_ts = self.fit(y_al, x_al)
        return SpreadSeries(y_al, x_al, beta_ts, estimator_name=self.__class__.__name__)


