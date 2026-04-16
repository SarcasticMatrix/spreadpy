import pandas as pd
import numpy as np

from typing import Tuple
import warnings

from spreadpy.data.priceTimeSeries import PriceTimeSeries

class SpreadSeries:
    """
    Residual spread series s_t = y_t − β_t · x_t.

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
        Ornstein-Uhlenbeck half-life via OLS regression of:
            Δspread_t = λ * spread_{t-1} + ε
        half_life = -ln(2) / λ
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
        """Returns (ADF test statistic, p-value). Requires statsmodels."""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(self._residuals.dropna(), maxlags=1)
            return float(result[0]), float(result[1])
        except ImportError:
            raise ImportError("statsmodels required for ADF test: pip install statsmodels")

    def rolling_zscore(self, window: int) -> pd.Series:
        """Z-score of spread over a rolling window (no lookahead)."""
        mu = self._residuals.rolling(window).mean()
        sigma = self._residuals.rolling(window).std()
        return ((self._residuals - mu) / sigma).rename("zscore")

    def slice(self, start, end) -> "SpreadSeries":
        """Return a SpreadSeries restricted to [start, end]."""
        mask = (self.index >= start) & (self.index <= end)
        sub_idx = self.index[mask]
        new_y = PriceTimeSeries(self.y.series.loc[sub_idx], name=self.y.name)
        new_x = PriceTimeSeries(self.x.series.loc[sub_idx], name=self.x.name)
        new_beta = self.hedge_ratio_ts.loc[sub_idx]
        return SpreadSeries(new_y, new_x, new_beta, self.estimator_name)
