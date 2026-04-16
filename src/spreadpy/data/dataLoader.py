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
    Loads price time series from CSV files or in-memory DataFrames.

    Convention: each CSV has a DatetimeIndex column + one or more price columns.

    Example usage
    -------------
    loader = DataLoader(base_path="data/")
    crack = loader.load_pair("crude_oil", "heating_oil", date_col="Date", price_col="Close")
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
        """Load a single asset by name (auto-detects csv/parquet/feather)."""
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
        """Load directly from an existing DataFrame."""
        df = df.set_index(date_col) if date_col in df.columns else df
        series = df[price_col].rename(name)
        return PriceTimeSeries(series)

    def load_from_series(self, series: pd.Series, name: str) -> PriceTimeSeries:
        return PriceTimeSeries(series, name=name)

    def load_pair(
        self,
        name_y: str,
        name_x: str,
        date_col: str = "Date",
        price_col: str = "Close",
        freq: Optional[str] = None,
    ) -> Tuple[PriceTimeSeries, PriceTimeSeries]:
        """Load and align two assets (y = dependent, x = independent leg)."""
        ts_y = self.load(name_y, date_col, price_col, freq)
        ts_x = self.load(name_x, date_col, price_col, freq)
        return ts_y.align(ts_x)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, ts: PriceTimeSeries, min_obs: int = 252) -> None:
        """Basic sanity checks on a loaded series."""
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
