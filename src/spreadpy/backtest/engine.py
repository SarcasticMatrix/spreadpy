"""
backtest/engine.py — Simple train / validation / test backtest engine
BacktestEngine
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from spreadpy.data import PriceTimeSeries
from spreadpy.spread import HedgeRatioEstimator, SpreadSeries
from spreadpy.signal import Direction, Signal, SignalGenerator
from spreadpy.sizing import PositionSizer
from spreadpy.backtest.costs import TransactionCosts
from spreadpy.backtest.portfolio import Portfolio
from spreadpy.backtest.metrics import RiskMetrics
from spreadpy.backtest.result import BacktestResult


class BacktestEngine:
    """
    Simple train / validation / test backtest engine.

    The series is split once into three consecutive periods:

        |<---- train_frac ---->|<-- val_frac -->|<-- test (remainder) -->|

    The hedge ratio estimator and signal generator are fitted on the
    training period only.  Use the validation result to tune
    hyper-parameters; the test result is the final held-out evaluation.

    Parameters
    ----------
    estimator        : HedgeRatioEstimator instance
    signal_gen       : SignalGenerator instance
    sizer            : PositionSizer instance
    costs            : TransactionCosts instance
    initial_capital  : Starting equity
    train_frac       : Fraction of data for training   (default 0.6)
    val_frac         : Fraction of data for validation  (default 0.2).
                       Set to 0.0 to skip validation entirely.
                       The remaining fraction is used as the test set.
    periods_per_year : For annualisation (252 = daily, 52 = weekly, etc.)
    """

    def __init__(
        self,
        estimator: HedgeRatioEstimator,
        signal_gen: SignalGenerator,
        sizer: PositionSizer,
        costs: Optional[TransactionCosts] = None,
        initial_capital: float = 1_000_000.0,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        periods_per_year: int = 252,
    ) -> None:
        if not 0.0 <= val_frac < 1.0:
            raise ValueError("val_frac must be in [0, 1)")
        if train_frac + val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0")
        self.estimator = estimator
        self.signal_gen = signal_gen
        self.sizer = sizer
        self.costs = costs or TransactionCosts()
        self.initial_capital = initial_capital
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.periods_per_year = periods_per_year

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        y: PriceTimeSeries,
        x: PriceTimeSeries,
    ) -> Tuple[Optional[BacktestResult], BacktestResult]:
        """
        Fit on train, evaluate on validation and test.

        Returns
        -------
        (val_result, test_result)
            val_result is None when val_frac=0.0.
        """
        y_al, x_al = y.align(x)
        train_idx, val_idx, test_idx = self._split(y_al)

        train_dates = y_al.index[train_idx]
        val_dates   = y_al.index[val_idx] if len(val_idx) else None
        test_dates  = y_al.index[test_idx]

        y_train = y_al.slice(train_dates[0], train_dates[-1])
        x_train = x_al.slice(train_dates[0], train_dates[-1])

        # Fit estimator on train; re-run over full range for Kalman continuity
        estimator = copy.deepcopy(self.estimator)
        estimator.fit(y_train, x_train)

        y_full  = y_al.slice(train_dates[0], test_dates[-1])
        x_full  = x_al.slice(train_dates[0], test_dates[-1])
        beta_full = estimator.fit(y_full, x_full)

        # Fit signal generator on the training spread
        beta_train  = beta_full.loc[train_dates[0]:train_dates[-1]]
        spread_train = SpreadSeries(
            y_train, x_train, beta_train,
            estimator_name=estimator.__class__.__name__,
        )
        signal_gen = copy.deepcopy(self.signal_gen)
        signal_gen.fit(spread_train)

        val_result = (
            self._run_split("val", val_dates, train_dates,
                            y_al, x_al, beta_full, signal_gen, estimator)
            if val_dates is not None else None
        )
        test_result = self._run_split("test", test_dates, train_dates,
                                      y_al, x_al, beta_full, signal_gen, estimator)

        return val_result, test_result

    # ------------------------------------------------------------------
    # Single split evaluation
    # ------------------------------------------------------------------

    def _run_split(
        self,
        split: str,
        eval_dates: pd.DatetimeIndex,
        train_dates: pd.DatetimeIndex,
        y_al: PriceTimeSeries,
        x_al: PriceTimeSeries,
        beta_full: pd.Series,
        signal_gen: SignalGenerator,
        estimator: HedgeRatioEstimator,
    ) -> BacktestResult:

        y_eval    = y_al.slice(eval_dates[0], eval_dates[-1])
        x_eval    = x_al.slice(eval_dates[0], eval_dates[-1])
        beta_eval = beta_full.loc[eval_dates[0]:eval_dates[-1]]

        spread = SpreadSeries(
            y_eval, x_eval, beta_eval,
            estimator_name=estimator.__class__.__name__,
        )
        signals = signal_gen.generate(spread)

        portfolio = Portfolio(self.initial_capital, self.costs)
        prev_direction = Direction.FLAT

        for i, (ts, sig) in enumerate(signals.items()):
            price_y     = float(y_eval.series.iloc[i])
            price_x     = float(x_eval.series.iloc[i])
            hedge_ratio = float(beta_eval.iloc[i]) if i < len(beta_eval) else 1.0

            qty_y, qty_x = self.sizer.size(sig, price_y, price_x, hedge_ratio)

            if sig.direction != prev_direction:
                portfolio.fill(ts, sig, qty_y, qty_x, price_y, price_x)
                prev_direction = sig.direction

            portfolio.mark(ts, price_y, price_x)

        # Close any remaining position at the end of the period
        last_py  = float(y_eval.series.iloc[-1])
        last_px  = float(x_eval.series.iloc[-1])
        flat_sig = Signal(Direction.FLAT, 0.0, eval_dates[-1])
        portfolio.fill(eval_dates[-1], flat_sig, 0.0, 0.0, last_py, last_px)
        portfolio.mark(eval_dates[-1], last_py, last_px)

        equity_curve = portfolio.equity_curve()
        metrics = RiskMetrics(equity_curve["equity"], risk_free_rate=0.0).summary(
            self.periods_per_year, trades=portfolio.trades
        )

        return BacktestResult(
            split=split,
            train_start=train_dates[0],  train_end=train_dates[-1],
            eval_start=eval_dates[0],    eval_end=eval_dates[-1],
            equity_curve=equity_curve,
            trades=portfolio.trades,
            signals=signals,
            spread=spread,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Split generation
    # ------------------------------------------------------------------

    def _split(
        self,
        y: PriceTimeSeries,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (train_idx, val_idx, test_idx) index arrays."""
        n         = len(y)
        train_end = int(n * self.train_frac)
        val_end   = int(n * (self.train_frac + self.val_frac))

        if train_end < 2 or n <= val_end:
            raise ValueError(
                f"Not enough data for the requested split "
                f"(n={n}, train_frac={self.train_frac}, val_frac={self.val_frac})."
            )

        return (
            np.arange(0, train_end),
            np.arange(train_end, val_end),  # empty array when val_frac=0
            np.arange(val_end, n),
        )
