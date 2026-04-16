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
        Estimate the hedge ratio series β_t from price series y and x.

        The returned series satisfies:

            s_t = y_t − β_t · x_t

        where s_t is (ideally) stationary. Implementations must ensure
        β_t is free of lookahead bias: at each bar t, β_t may only depend
        on observations {(y_s, x_s) : s ≤ t}.

        :param PriceTimeSeries y: Dependent-leg price series.
        :param PriceTimeSeries x: Independent-leg price series.
        :returns: Time series of hedge ratios aligned with ``y.index``.
        :rtype: pd.Series
        """

    def compute_spread(self, y: PriceTimeSeries, x: PriceTimeSeries) -> "SpreadSeries":
        """
        Fit the estimator and return the residual spread in a single call.

        Equivalent to ``SpreadSeries(y, x, self.fit(y, x))``. The spread
        is defined bar-by-bar as:

            s_t = y_t − β_t · x_t

        :param PriceTimeSeries y: Dependent-leg price series.
        :param PriceTimeSeries x: Independent-leg price series.
        :returns: Residual spread series with diagnostics.
        :rtype: SpreadSeries
        """
        y_al, x_al = y.align(x)
        beta_ts = self.fit(y_al, x_al)
        return SpreadSeries(y_al, x_al, beta_ts, estimator_name=self.__class__.__name__)


