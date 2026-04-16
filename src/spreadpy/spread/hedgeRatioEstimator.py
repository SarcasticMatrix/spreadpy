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
    Abstract base class for hedge ratio estimators.

    Given two price series y (dependent leg) and x (independent leg),
    an estimator produces a time series β_t such that the spread

        s_t = y_t − β_t · x_t

    is (ideally) stationary. All concrete subclasses must implement
    :meth:`fit`, which returns a ``pd.Series`` of hedge ratios aligned
    with ``y.index``.
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


