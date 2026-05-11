"""
data.py — Layer 1: Data primitives
PriceTimeSeries, DataLoader, TransactionCosts
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from spreadpy.data.priceTimeSeries import PriceTimeSeries

class DataLoader:
    """
    Loads price time series from CSV, Parquet, or Feather files.

    File lookup is by name: ``base_path/<name>.<ext>`` where ``<ext>``
    is tried in order ``.csv``, ``.parquet``, ``.feather``. All loaded
    series are returned as :class:`PriceTimeSeries` objects.

    :param Union[str, Path] base_path: Root directory that contains the data files.
    """

    SUPPORTED_FORMATS = {".csv", ".parquet", ".feather"}

    def __init__(self, base_path: Union[str, Path] = ".") -> None:
        self.base_path = Path(base_path)

    # ------------------------------------------------------------------
    # Core loading
    # ------------------------------------------------------------------

    def load(
        self,
        name: str,
        date_col: str = "Date",
        price_col: str = "Close",
        freq: Optional[str] = None,
    ) -> PriceTimeSeries:
        """Load a single asset by name, auto-detecting csv / parquet / feather.

        The file is searched under ``base_path/<name>.<ext>`` where ``<ext>``
        is tried in order ``.csv``, ``.parquet``, ``.feather``.

        :param str name: Asset identifier used as the file stem and series name.
        :param str date_col: Column (or index) that contains the timestamps.
        :param str price_col: Column that contains the price series.
        :param Optional[str] freq: If given, the series is resampled to this
            pandas offset alias (e.g. ``'W'``, ``'ME'``) using the last price.

        :returns: Cleaned price series for the asset.
        :rtype: PriceTimeSeries
        :raises FileNotFoundError: If no file matching ``name`` is found.
        """
        path = self._find_file(name)
        df = self._read_file(path, date_col)
        series = df[price_col].rename(name)
        ts = PriceTimeSeries(series)
        if freq:
            ts = ts.resample(freq)
        return ts

    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        date_col: str = "Date",
        price_col: str = "Close",
    ) -> PriceTimeSeries:
        """Load a price series directly from an existing DataFrame.

        If ``date_col`` is a column of ``df``, it is set as the index.
        Otherwise ``df.index`` is assumed to already be the DatetimeIndex.

        :param pd.DataFrame df: Source DataFrame.
        :param str name: Label assigned to the resulting series.
        :param str date_col: Column name holding timestamps (ignored if already
            the index).
        :param str price_col: Column name holding the price series.

        :returns: Cleaned price series.
        :rtype: PriceTimeSeries
        """
        df = df.set_index(date_col) if date_col in df.columns else df
        series = df[price_col].rename(name)
        return PriceTimeSeries(series)

    def load_from_series(self, series: pd.Series, name: str) -> PriceTimeSeries:
        """Wrap an existing pandas Series as a :class:`PriceTimeSeries`.

        :param pd.Series series: Raw price series with a DatetimeIndex (or
            an index coercible to DatetimeIndex).
        :param str name: Label assigned to the series.

        :returns: Cleaned price series.
        :rtype: PriceTimeSeries
        """
        return PriceTimeSeries(series, name=name)

    def load_pair(
        self,
        name_y: str,
        name_x: str,
        date_col: str = "Date",
        price_col: str = "Close",
        freq: Optional[str] = None,
    ) -> Tuple[PriceTimeSeries, PriceTimeSeries]:
        """Load two assets and align them on their common timestamps.

        Each asset is loaded via :meth:`load` then the two series are inner-joined
        on their DatetimeIndex, so the returned pair shares an identical index
        with no missing observations.

        :param str name_y: File stem for the dependent leg y.
        :param str name_x: File stem for the independent leg x.
        :param str date_col: Column (or index) that contains the timestamps.
        :param str price_col: Column that contains the price series.
        :param Optional[str] freq: If given, each series is resampled before
            alignment.

        :returns: Aligned pair ``(ts_y, ts_x)`` sharing a common DatetimeIndex.
        :rtype: Tuple[PriceTimeSeries, PriceTimeSeries]
        """
        ts_y = self.load(name_y, date_col, price_col, freq)
        ts_x = self.load(name_x, date_col, price_col, freq)
        return ts_y.align(ts_x)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, ts: PriceTimeSeries, min_obs: int = 252) -> None:
        """Run basic sanity checks on a loaded series.

        Raises :class:`ValueError` if any of the following conditions hold:

        - Fewer than ``min_obs`` observations.
        - Non-positive prices (prices ≤ 0).
        - Duplicate timestamps.

        :param PriceTimeSeries ts: Series to validate.
        :param int min_obs: Minimum number of observations required (default 252).

        :raises ValueError: If any sanity check fails.
        """
        if len(ts) < min_obs:
            raise ValueError(
                f"{ts.name}: only {len(ts)} observations (min={min_obs})"
            )
        if (ts.values <= 0).any():
            raise ValueError(f"{ts.name}: non-positive prices detected")
        dup = ts.index[ts.index.duplicated()]
        if len(dup):
            raise ValueError(f"{ts.name}: duplicate timestamps {dup[:3].tolist()}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_file(self, name: str) -> Path:
        for ext in self.SUPPORTED_FORMATS:
            candidate = self.base_path / f"{name}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No file found for '{name}' in {self.base_path} "
            f"(tried {self.SUPPORTED_FORMATS})"
        )

    def _read_file(self, path: Path, date_col: str) -> pd.DataFrame:
        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
            if date_col in df.columns:
                df = df.set_index(date_col)
            df.index = pd.to_datetime(df.index)
        elif ext == ".feather":
            df = pd.read_feather(path)
            if date_col in df.columns:
                df = df.set_index(date_col)
            df.index = pd.to_datetime(df.index)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        return df
