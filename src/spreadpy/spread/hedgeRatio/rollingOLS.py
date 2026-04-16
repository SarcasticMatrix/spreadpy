from typing import Optional

import numpy as np
import pandas as pd

from spreadpy.data import PriceTimeSeries
from spreadpy.spread.hedgeRatioEstimator import HedgeRatioEstimator

class RollingOLS(HedgeRatioEstimator):
    """
    Rolling OLS hedge ratio estimator.

    At each bar t, fits OLS on the most recent ``window`` observations:

        y_s = α + β_t · x_s + ε_s,   s ∈ [t − w + 1,  t]

    This adapts to structural shifts in the cointegration relationship
    without the complexity of a state-space model. Bars before the first
    complete window are filled with the first available estimate (forward-fill).

    :param int window: Number of bars in each rolling regression (must be ≥ 2).
    :param bool add_intercept: If True (default), includes an intercept term α.
    """

    def __init__(self, window: int = 60, add_intercept: bool = True) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.add_intercept = add_intercept

    def fit(self, y: PriceTimeSeries, x: PriceTimeSeries) -> pd.Series:
        """
        Estimate a time-varying hedge ratio β_t via rolling OLS.

        At each bar t ≥ w − 1, the hedge ratio is the OLS slope over the
        most recent w observations:

            β_t = argmin_{β,α} Σ_{s=t−w+1}^{t} (y_s − α − β·x_s)²

        whose solution is:

            β_t = Cov_w[x, y] / Var_w[x]

        where Cov_w and Var_w denote the sample covariance and variance
        over the rolling window. For the first w − 1 bars (warm-up), the
        first available estimate is forward-filled.

        :param PriceTimeSeries y: Dependent-leg price series.
        :param PriceTimeSeries x: Independent-leg price series.
        :returns: Time series of hedge ratios β_t aligned with ``y.index``.
        :rtype: pd.Series
        """
        y_al, x_al = y.align(x)
        yv, xv = y_al.values.astype(float), x_al.values.astype(float)
        n = len(yv)
        betas = np.full(n, np.nan)

        for t in range(self.window - 1, n):
            y_w = yv[t - self.window + 1 : t + 1]
            x_w = xv[t - self.window + 1 : t + 1]
            if self.add_intercept:
                X = np.column_stack([x_w, np.ones(self.window)])
            else:
                X = x_w.reshape(-1, 1)
            try:
                coef, *_ = np.linalg.lstsq(X, y_w, rcond=None)
                betas[t] = coef[0]
            except np.linalg.LinAlgError:
                pass

        beta_series = pd.Series(betas, index=y_al.index, name="hedge_ratio")
        # Forward-fill the warm-up period
        return beta_series.ffill().bfill()
