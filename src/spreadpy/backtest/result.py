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
    Immutable container for the output of a single backtest split.

    Produced by :class:`BacktestEngine` for both the validation and test
    periods. The ``split`` field identifies which period this result covers.

    :param str split: Period identifier — ``'val'`` or ``'test'``.
    :param pd.Timestamp train_start: Start of the in-sample (fitting) period.
    :param pd.Timestamp train_end: End of the in-sample (fitting) period.
    :param pd.Timestamp eval_start: Start of the out-of-sample evaluation period.
    :param pd.Timestamp eval_end: End of the out-of-sample evaluation period.
    :param pd.DataFrame equity_curve: Bar-level mark-to-market equity.
        Columns: ``equity``, ``cash``, ``unrealised_pnl``, ``realised_pnl``, ``total_costs``.
    :param List[Trade] trades: All leg fills executed during the eval period.
    :param pd.Series signals: :class:`Signal` objects indexed by timestamp.
    :param SpreadSeries spread: Spread series for the eval period.
    :param pd.Series metrics: Risk metrics summary from :class:`RiskMetrics`.
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
