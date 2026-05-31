from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class PriceTimeSeries:
    """
    Thin wrapper around a pandas Series for price data.

    Guarantees a clean, well-formed series at construction time:
    sorted DatetimeIndex, no NaN values, and a consistent name.
    NaNs are either forward-filled or dropped depending on ``fill_method``.

    :param pd.Series prices: Raw price series. A non-DatetimeIndex is coerced
        to DatetimeIndex automatically.
    :param str name: Optional label for the series (used in repr and downstream labelling).
    :param str fill_method: NaN handling strategy — ``'ffill'`` (forward-fill, then drop
        any leading NaNs) or ``'drop'`` (remove all NaN rows entirely).
    """

    def __init__(
        self,
        prices: pd.Series,
        name: Optional[str] = None,
        fill_method: str = "ffill",
    ) -> None:
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)

        prices = prices.sort_index()

        if fill_method == "ffill":
            prices = prices.ffill().dropna()
        elif fill_method == "drop":
            prices = prices.dropna()
        else:
            raise ValueError(f"Unknown fill_method: {fill_method}")

        self._series: pd.Series = prices.rename(name or prices.name or "price")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def series(self) -> pd.Series:
        return self._series

    @property
    def name(self) -> str:
        return str(self._series.name)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._series.index  # type: ignore[return-value]

    @property
    def values(self) -> np.ndarray:
        return self._series.values

    def __len__(self) -> int:
        return len(self._series)

    def __repr__(self) -> str:
        return (
            f"PriceTimeSeries(name={self.name!r}, "
            f"n={len(self)}, "
            f"start={self.index[0].date()}, "
            f"end={self.index[-1].date()})"
        )

    # ------------------------------------------------------------------
    # Alignment / manipulation
    # ------------------------------------------------------------------

    def align(self, other: "PriceTimeSeries") -> Tuple["PriceTimeSeries", "PriceTimeSeries"]:
        """Inner-join two series on their common timestamps.

        Only observations present in **both** series are kept.  The resulting
        pair shares an identical DatetimeIndex with no missing values.

        :param PriceTimeSeries other: Second price series to align with.

        :returns: Pair ``(self_aligned, other_aligned)`` restricted to the
            intersection of their DatetimeIndex.
        :rtype: Tuple[PriceTimeSeries, PriceTimeSeries]
        """
        s1, s2 = self._series.align(other._series, join="inner")
        return (
            PriceTimeSeries(s1, name=self.name),
            PriceTimeSeries(s2, name=other.name),
        )

    def resample(self, freq: str) -> "PriceTimeSeries":
        """Resample to a lower frequency, keeping the last price in each period.

        Uses ``pd.Series.resample(freq).last()`` then drops any resulting NaN
        bins (e.g. empty periods).

        :param str freq: Pandas offset alias for the target frequency
            (e.g. ``'W'`` for weekly, ``'ME'`` for month-end).

        :returns: Resampled price series.
        :rtype: PriceTimeSeries
        """
        resampled = self._series.resample(freq).last().dropna()
        return PriceTimeSeries(resampled, name=self.name)

    def log_returns(self) -> pd.Series:
        """Compute continuously compounded log-returns.

        The log-return at bar :math:`t` is defined as:

        .. math::

            r_t = \\log\\!\\left(\\frac{P_t}{P_{t-1}}\\right)

        The first observation is dropped (NaN from the lag).

        :returns: Log-return series aligned with ``self.index[1:]``.
        :rtype: pd.Series
        """
        return np.log(self._series / self._series.shift(1)).dropna()

    def slice(self, start, end) -> "PriceTimeSeries":
        """Extract a sub-series over the closed interval [start, end].

        :param start: Inclusive start timestamp (``pd.Timestamp`` or any value
            accepted by pandas label-based indexing).
        :param end: Inclusive end timestamp.

        :returns: Price sub-series restricted to [start, end].
        :rtype: PriceTimeSeries
        """
        return PriceTimeSeries(self._series.loc[start:end], name=self.name)

    def returns(self) -> pd.Series:
        """Compute simple (arithmetic) returns.

        The return at bar :math:`t` is defined as:

        .. math::

            r_t = \\frac{P_t - P_{t-1}}{P_{t-1}}

        The first observation is dropped (NaN from the lag).

        :returns: Simple return series aligned with ``self.index[1:]``.
        :rtype: pd.Series
        """
        return self._series.pct_change().dropna()

