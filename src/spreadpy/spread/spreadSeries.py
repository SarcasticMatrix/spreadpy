import pandas as pd
import numpy as np

from typing import Tuple
import warnings

from spreadpy.data.priceTimeSeries import PriceTimeSeries

class SpreadSeries:
    """
    Residual spread series :math:`s_t = y_t - \\beta_t \\cdot x_t`.

    Given aligned price series y, x and a time-varying hedge ratio β_t,
    the spread is computed bar-by-bar. Provides stationarity diagnostics
    (Ornstein-Uhlenbeck half-life, ADF test) and a rolling z-score.

    :param PriceTimeSeries y: Dependent leg price series.
    :param PriceTimeSeries x: Independent leg price series.
    :param pd.Series hedge_ratio_ts: Time series of hedge ratios β_t,
        aligned (or reindexed + forward-filled) to ``y.index``.
    :param str estimator_name: Label for the estimator used (for display and repr).
    """

    def __init__(
        self,
        y: PriceTimeSeries,
        x: PriceTimeSeries,
        hedge_ratio_ts: pd.Series,
        estimator_name: str = "",
    ) -> None:
        y_al, x_al = y.align(x)
        self.y = y_al
        self.x = x_al
        self.hedge_ratio_ts = hedge_ratio_ts.reindex(y_al.index).ffill()
        self.estimator_name = estimator_name

        self._residuals: pd.Series = (
            y_al.series - self.hedge_ratio_ts * x_al.series
        ).rename("spread")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def residuals(self) -> pd.Series:
        return self._residuals

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._residuals.index  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self._residuals)

    def __repr__(self) -> str:
        return (
            f"SpreadSeries(estimator={self.estimator_name!r}, "
            f"n={len(self)}, "
            f"mean={self._residuals.mean():.4f}, "
            f"std={self._residuals.std():.4f})"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def half_life(self) -> float:
        """
        Estimate the mean-reversion half-life of the spread via an
        Ornstein-Uhlenbeck OLS regression.

        The discrete-time OU model is:

        .. math::

            \\Delta s_t = \\lambda \\cdot s_{t-1} + c + \\varepsilon_t,
            \\quad \\varepsilon_t \\sim \\mathcal{N}(0, \\sigma^2)

        where :math:`\\lambda < 0` indicates mean-reversion. The half-life
        is the time (in bars) for a deviation from equilibrium to decay by half:

        .. math::

            \\tau_{1/2} = -\\frac{\\ln 2}{\\lambda}

        :math:`\\lambda` is estimated by regressing :math:`\\Delta s_t` on
        :math:`s_{t-1}` (with constant) using OLS. If :math:`\\lambda \\geq 0`,
        the spread is non-stationary and the method returns ``float('inf')``
        with a warning.

        :returns: Mean-reversion half-life in bars. ``float('inf')`` if the
            spread does not mean-revert.
        :rtype: float
        """
        spread = self._residuals.dropna()
        delta = spread.diff().dropna()
        lagged = spread.shift(1).dropna()
        delta, lagged = delta.align(lagged, join="inner")

        slope, *_ = np.linalg.lstsq(
            np.column_stack([lagged, np.ones(len(lagged))]),
            delta.values,
            rcond=None,
        )
        lam = slope[0]
        if lam >= 0:
            warnings.warn(
                "Positive mean-reversion coefficient — spread may not be stationary",
                stacklevel=2,
            )
            return float("inf")
        return -np.log(2) / lam

    def adf_statistic(self) -> Tuple[float, float]:
        """
        Compute the Augmented Dickey-Fuller test statistic for the spread.

        Tests the null hypothesis :math:`H_0` of a unit root (non-stationarity)
        against the alternative :math:`H_1` of stationarity (mean-reversion).
        The ADF regression with one lag is:

        .. math::

            \\Delta s_t = \\rho \\cdot s_{t-1} + c + \\delta \\cdot \\Delta s_{t-1} + \\varepsilon_t

        A t-statistic on :math:`\\rho` that is sufficiently negative
        (p-value < 0.05) rejects :math:`H_0` and supports cointegration of
        the pair.

        Requires ``statsmodels``.

        :returns: Tuple ``(adf_statistic, p_value)``.
        :rtype: Tuple[float, float]
        :raises ImportError: If ``statsmodels`` is not installed.
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(self._residuals.dropna())
            return float(result[0]), float(result[1])
        except ImportError:
            raise ImportError("statsmodels required for ADF test: pip install statsmodels")

    def rolling_zscore(self, window: int) -> pd.Series:
        """
        Compute a rolling z-score of the spread with no lookahead.

        At each bar :math:`t`, the z-score standardises :math:`s_t` using
        statistics estimated over the preceding window of :math:`w` bars:

        .. math::

            \\mu_{t,w} &= \\frac{1}{w} \\sum_{j=0}^{w-1} s_{t-j} \\\\
            \\sigma_{t,w} &= \\mathrm{std}(s_{t-w+1}, \\ldots, s_t) \\quad (\\mathrm{ddof}=1) \\\\
            z_t &= \\frac{s_t - \\mu_{t,w}}{\\sigma_{t,w}}

        Bars :math:`t < w` are set to NaN (insufficient history). This is the
        signal used by :class:`ZScoreSignal`.

        :param int window: Rolling window length w (in bars).
        :returns: z-score series aligned with ``self.index``.
        :rtype: pd.Series
        """
        mu = self._residuals.rolling(window).mean()
        sigma = self._residuals.rolling(window).std()
        return ((self._residuals - mu) / sigma).rename("zscore")

    def slice(self, start, end) -> "SpreadSeries":
        """
        Return a new :class:`SpreadSeries` restricted to the closed
        interval [start, end].

        The underlying y, x, and β_t series are all sliced consistently,
        so the returned object is a self-contained SpreadSeries over the
        sub-period with no reference to the original data outside [start, end].

        :param start: Inclusive start timestamp (pd.Timestamp or compatible).
        :param end: Inclusive end timestamp (pd.Timestamp or compatible).
        :returns: Spread sub-series over [start, end].
        :rtype: SpreadSeries
        """
        mask = (self.index >= start) & (self.index <= end)
        sub_idx = self.index[mask]
        new_y = PriceTimeSeries(self.y.series.loc[sub_idx], name=self.y.name)
        new_x = PriceTimeSeries(self.x.series.loc[sub_idx], name=self.x.name)
        new_beta = self.hedge_ratio_ts.loc[sub_idx]
        return SpreadSeries(new_y, new_x, new_beta, self.estimator_name)
