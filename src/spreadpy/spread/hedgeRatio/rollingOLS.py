from typing import Optional

import numpy as np
import pandas as pd

from spreadpy.data import PriceTimeSeries
from spreadpy.spread.hedgeRatioEstimator import HedgeRatioEstimator

class RollingOLS(HedgeRatioEstimator):
    """
    Rolling OLS hedge ratio estimation.

    At each bar t, OLS is run on the window [t-window+1, t].
    Bars before the first full window are filled with the first
    available estimate (forward-fill).
    """

    def __init__(self, window: int = 60, add_intercept: bool = True) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.add_intercept = add_intercept

    def fit(self, y: PriceTimeSeries, x: PriceTimeSeries) -> pd.Series:
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
