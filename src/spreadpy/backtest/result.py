"""
backtest/result.py — Backtest output container
BacktestResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from spreadpy.backtest.portfolio import Trade
from spreadpy.spread import SpreadSeries
from spreadpy.signal import Signal


@dataclass
class BacktestResult:
    """
    Output of a single backtest split (validation or test).

    Attributes
    ----------
    split                          : "val" or "test"
    train_start, train_end         : In-sample (fitting) period
    eval_start, eval_end           : Out-of-sample evaluation period
    equity_curve                   : Mark-to-market equity over the eval period
    trades                         : All fills in the eval period
    signals                        : Full signal series over the eval period
    spread                         : SpreadSeries for the eval period
    metrics                        : Risk metrics summary (pd.Series)
    """

    split: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    equity_curve: pd.DataFrame
    trades: List[Trade]
    signals: pd.Series
    spread: SpreadSeries
    metrics: pd.Series

    def __repr__(self) -> str:
        sr  = self.metrics.get("sharpe", float("nan"))
        mdd = self.metrics.get("max_drawdown", float("nan"))
        return (
            f"BacktestResult(split={self.split!r}, "
            f"eval=[{self.eval_start.date()}→{self.eval_end.date()}], "
            f"sharpe={sr:.2f}, mdd={mdd:.1%})"
        )
