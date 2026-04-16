"""
utils.py — Data fetching utilities.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_history(
    ticker: str,
    period: str = "730d",
    interval: str = "1h",
) -> pd.Series:
    """
    Download intraday price history from Yahoo Finance.

    Yahoo Finance caps hourly data at ~730 calendar days. For longer
    histories use ``interval="1d"`` (no hard cap).

    :param str ticker: Yahoo Finance ticker symbol (e.g. ``"CL=F"``).
    :param str period: Lookback period string accepted by yfinance
        (default ``"730d"``).
    :param str interval: Bar interval string accepted by yfinance
        (default ``"1h"``).
    :returns: Close prices with a UTC-aware DatetimeIndex, sorted and
        deduplicated.
    :rtype: pd.Series
    """
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker!r}")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close[ticker]

    if close.index.tz is None:
        close.index = close.index.tz_localize("UTC")
    else:
        close.index = close.index.tz_convert("UTC")

    close = close.sort_index()
    close = close[~close.index.duplicated(keep="first")]
    close.name = ticker
    return close
