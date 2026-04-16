from typing import Optional

import numpy as np
import pandas as pd

from spreadpy.data import PriceTimeSeries
from spreadpy.spread.hedgeRatioEstimator import HedgeRatioEstimator

class ConstantOLS(HedgeRatioEstimator):
    """
    Single full-sample OLS regression.
    β is constant across the entire series.
    """

    def __init__(self, add_intercept: bool = True) -> None:
        self.add_intercept = add_intercept
        self.beta_: Optional[float] = None
        self.alpha_: Optional[float] = None
        self.r_squared_: Optional[float] = None

    def fit(self, y: PriceTimeSeries, x: PriceTimeSeries) -> pd.Series:
        y_al, x_al = y.align(x)
        yv, xv = y_al.values, x_al.values

        if self.add_intercept:
            X = np.column_stack([xv, np.ones(len(xv))])
            result = np.linalg.lstsq(X, yv, rcond=None)
            self.beta_, self.alpha_ = result[0][0], result[0][1]
        else:
            self.beta_ = float(np.dot(xv, yv) / np.dot(xv, xv))
            self.alpha_ = 0.0

        y_hat = self.beta_ * xv + (self.alpha_ or 0.0)
        ss_res = np.sum((yv - y_hat) ** 2)
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        self.r_squared_ = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return pd.Series(self.beta_, index=y_al.index, name="hedge_ratio")
