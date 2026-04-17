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
from spreadpy.sizing import LinearSizer
from spreadpy.backtest.costs import TransactionCosts
from spreadpy.backtest.portfolio import Portfolio
from spreadpy.backtest.metrics import RiskMetrics
from spreadpy.backtest.result import BacktestResult


class BacktestEngine:
    """
    Simple train / validation / test backtest engine.

    Splits the data once into three consecutive periods and fits the full
    pipeline — hedge ratio estimator then signal generator — on the training
    period only:

        |←── train_frac ──→|←── val_frac ──→|←── test (remainder) ──→|

    Use the validation result to tune hyperparameters; consult the test
    result only for the final held-out evaluation.

    :param HedgeRatioEstimator estimator: Hedge ratio estimator
        (e.g. :class:`KalmanFilterWithVelocity`).
    :param SignalGenerator signal_gen: Signal generator
        (e.g. :class:`ZScoreSignal`).
    :param LinearSizer sizer: Position sizing model.
    :param Optional[TransactionCosts] costs: Transaction cost model.
        Defaults to 2 bps slippage + 1 bps commission.
    :param float initial_capital: Starting equity in monetary units.
    :param float train_frac: Fraction of data reserved for training / fitting
        (default 0.6).
    :param float val_frac: Fraction of data used as validation set (default 0.2).
        Set to 0.0 for a simple train / test split with no validation —
        :meth:`run` will then return ``(None, test_result)``.
        The remainder ``1 − train_frac − val_frac`` forms the test set.
    :param int periods_per_year: Bars per year for annualisation
        (252 for daily, 252 × 23 ≈ 5796 for hourly futures).
    :param bool log_prices: If ``True``, log-transform prices before passing
        them to the hedge ratio estimator and signal generator. The Kalman
        filter then operates on log-prices, which better satisfies the
        constant observation-noise assumption (σ²_ε homoscedastic).
        Position sizing and P&L accounting always use the original prices.
    """

    def __init__(
        self,
        estimator: HedgeRatioEstimator,
        signal_gen: SignalGenerator,
        sizer: LinearSizer,
        costs: Optional[TransactionCosts] = None,
        initial_capital: float = 1_000_000.0,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        periods_per_year: int = 252,
        log_prices: bool = False,
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
        self.log_prices = log_prices

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

        # Signal series: optionally log-transformed for Kalman / spread / z-score.
        # Trading series: always original prices for sizing and P&L.
        y_sig = self._signal_series(y_al)
        x_sig = self._signal_series(x_al)

        y_train_sig = y_sig.slice(train_dates[0], train_dates[-1])
        x_train_sig = x_sig.slice(train_dates[0], train_dates[-1])

        # Fit estimator on train; re-run over full range for Kalman continuity
        estimator = copy.deepcopy(self.estimator)
        estimator.fit(y_train_sig, x_train_sig)

        y_full_sig = y_sig.slice(train_dates[0], test_dates[-1])
        x_full_sig = x_sig.slice(train_dates[0], test_dates[-1])
        beta_full  = estimator.fit(y_full_sig, x_full_sig)

        # Fit signal generator on the training spread
        beta_train   = beta_full.loc[train_dates[0]:train_dates[-1]]
        spread_train = SpreadSeries(
            y_train_sig, x_train_sig, beta_train,
            estimator_name=estimator.__class__.__name__,
        )
        signal_gen = copy.deepcopy(self.signal_gen)
        signal_gen.fit(spread_train)

        val_result = (
            self._run_split("val", val_dates, train_dates,
                            y_al, x_al, y_sig, x_sig, beta_full, signal_gen, estimator)
            if val_dates is not None else None
        )
        test_result = self._run_split("test", test_dates, train_dates,
                                      y_al, x_al, y_sig, x_sig, beta_full, signal_gen, estimator)

        return val_result, test_result

    # ------------------------------------------------------------------
    # Single split evaluation
    # ------------------------------------------------------------------

    def _run_split(
        self,
        split: str,
        eval_dates: pd.DatetimeIndex,
        train_dates: pd.DatetimeIndex,
        y_al: PriceTimeSeries,       # original prices  — trading / P&L
        x_al: PriceTimeSeries,
        y_sig_al: PriceTimeSeries,   # signal prices    — Kalman / spread
        x_sig_al: PriceTimeSeries,
        beta_full: pd.Series,
        signal_gen: SignalGenerator,
        estimator: HedgeRatioEstimator,
    ) -> BacktestResult:

        y_eval     = y_al.slice(eval_dates[0], eval_dates[-1])
        x_eval     = x_al.slice(eval_dates[0], eval_dates[-1])
        y_eval_sig = y_sig_al.slice(eval_dates[0], eval_dates[-1])
        x_eval_sig = x_sig_al.slice(eval_dates[0], eval_dates[-1])
        beta_eval  = beta_full.loc[eval_dates[0]:eval_dates[-1]]

        spread = SpreadSeries(
            y_eval_sig, x_eval_sig, beta_eval,
            estimator_name=estimator.__class__.__name__,
        )
        signals = signal_gen.generate(spread)

        portfolio = Portfolio(self.initial_capital, self.costs)
        prev_direction = Direction.FLAT

        for i, (ts, sig) in enumerate(signals.items()):
            price_y     = float(y_eval.series.iloc[i])    # actual price for trading
            price_x     = float(x_eval.series.iloc[i])
            hedge_ratio = float(beta_eval.iloc[i]) if i < len(beta_eval) else 1.0

            qty_y, qty_x = self.sizer.size(sig, price_y, price_x, hedge_ratio,
                                           capital=portfolio.current_equity)

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

    def _signal_series(self, pts: PriceTimeSeries) -> PriceTimeSeries:
        """Return pts log-transformed when log_prices=True, else pts unchanged."""
        if self.log_prices:
            return PriceTimeSeries(np.log(pts.series), name=pts.name)
        return pts

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
