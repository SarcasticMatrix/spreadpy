from typing import Optional

import numpy as np
import pandas as pd

from spreadpy.data import PriceTimeSeries
from spreadpy.spread.hedgeRatioEstimator import HedgeRatioEstimator

class ConstantOLS(HedgeRatioEstimator):
    """
    Full-sample OLS hedge ratio estimator.

    Fits a single linear regression over the entire supplied period:

    .. math::

        y_t = \\alpha + \\beta \\cdot x_t + \\varepsilon_t

    and returns the constant slope :math:`\\beta` as the hedge ratio for all bars.
    Suitable as a baseline when the cointegration relationship is stable.

    After calling :meth:`fit`, the fitted values are available as
    ``beta_``, ``alpha_``, and ``r_squared_``.

    :param bool add_intercept: If True (default), fits with an intercept :math:`\\alpha`.
        If False, forces the regression through the origin (:math:`\\alpha = 0`).
    """

    def __init__(self, add_intercept: bool = True) -> None:
        self.add_intercept = add_intercept
        self.beta_: Optional[float] = None
        self.alpha_: Optional[float] = None
        self.r_squared_: Optional[float] = None

    def fit(self, y: PriceTimeSeries, x: PriceTimeSeries) -> pd.Series:
        """
        Estimate a single hedge ratio :math:`\\beta` via full-sample OLS and
        return it as a constant series.

        **With intercept** (default): solves the normal equations

        .. math::

            [\\beta,\\, \\alpha]^\\top = (X^\\top X)^{-1} X^\\top y,
            \\quad X = [x \\mid \\mathbf{1}]

        **Without intercept**: uses the closed-form projection

        .. math::

            \\beta = \\frac{x^\\top y}{x^\\top x}

        After fitting, ``beta_``, ``alpha_``, and ``r_squared_`` are set.
        The coefficient of determination is:

        .. math::

            R^2 = 1 - \\frac{SS_{\\mathrm{res}}}{SS_{\\mathrm{tot}}}, \\quad
            SS_{\\mathrm{res}} = \\|y - \\hat{\\beta}\\, x - \\hat{\\alpha}\\|^2, \\quad
            SS_{\\mathrm{tot}} = \\|y - \\bar{y}\\|^2

        :param PriceTimeSeries y: Dependent-leg price series.
        :param PriceTimeSeries x: Independent-leg price series.
        :returns: Constant hedge ratio :math:`\\beta` broadcast over ``y.index``.
        :rtype: pd.Series
        """
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
