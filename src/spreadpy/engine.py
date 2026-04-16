"""
engine.py — Layer 5: Walk-forward backtest engine
BacktestResult, WalkForwardEngine
"""

from __future__ import annotations

import copy
import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from spreadpy.data import PriceTimeSeries, TransactionCosts
from spreadpy.spread import HedgeRatioEstimator, SpreadSeries
from spreadpy.signal import Direction, Signal, SignalGenerator
from spreadpy.sizing import PositionSizer
from spreadpy.portfolio import RiskMetrics, Trade, Portfolio

@dataclass
class BacktestResult:
    """
    Output of a single walk-forward fold.

    Attributes
    ----------
    fold_id       : Index of this fold (0-based)
    train_start, train_end : In-sample period
    test_start, test_end   : Out-of-sample period
    equity_curve  : Mark-to-market equity over the test period
    trades        : All fills in the test period
    signals       : Full signal series over the test period
    spread        : SpreadSeries for the test period
    metrics       : Risk metrics summary (pd.Series)
    params        : Parameter dict used in this fold
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    equity_curve: pd.DataFrame
    trades: List[Trade]
    signals: pd.Series
    spread: SpreadSeries
    metrics: pd.Series
    params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        sr = self.metrics.get("sharpe", float("nan"))
        mdd = self.metrics.get("max_drawdown", float("nan"))
        return (
            f"BacktestResult(fold={self.fold_id}, "
            f"test=[{self.test_start.date()}→{self.test_end.date()}], "
            f"sharpe={sr:.2f}, mdd={mdd:.1%})"
        )

class WalkForwardEngine:
    """
    Orchestrates a walk-forward backtest over a pair of price series.

    Walk-forward scheme
    -------------------
    The series is split into consecutive folds:

        |<--- train_size --->|<--- test_size --->|
                             |<--- train_size --->|<--- test_size --->|
                                                  ...

    At each fold:
      1. Fit HedgeRatioEstimator on the training window.
      2. Build SpreadSeries for both train and test periods.
      3. Fit SignalGenerator on the training spread.
      4. Generate signals on the test spread (no lookahead).
      5. Run the portfolio loop: size, fill, mark-to-market.
      6. Compute risk metrics.

    Parameters
    ----------
    loader          : DataLoader (optional, can pass series directly)
    estimator       : HedgeRatioEstimator instance
    signal_gen      : SignalGenerator instance
    sizer           : PositionSizer instance
    costs           : TransactionCosts instance
    initial_capital : Starting equity for each fold (resets between folds)
    train_size      : Number of bars for training
    test_size       : Number of bars for testing
    step_size       : How many bars to advance the window each fold.
                      Default = test_size (non-overlapping folds).
    periods_per_year: For annualisation (252 = daily, 52 = weekly, etc.)
    """

    def __init__(
        self,
        estimator: HedgeRatioEstimator,
        signal_gen: SignalGenerator,
        sizer: PositionSizer,
        costs: Optional[TransactionCosts] = None,
        initial_capital: float = 1_000_000.0,
        train_size: int = 252,
        test_size: int = 63,
        step_size: Optional[int] = None,
        periods_per_year: int = 252,  # For 1-minute data
    ) -> None:
        self.estimator = estimator
        self.signal_gen = signal_gen
        self.sizer = sizer
        self.costs = costs or TransactionCosts()
        self.initial_capital = initial_capital
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.periods_per_year = periods_per_year

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run(
        self,
        y: PriceTimeSeries,
        x: PriceTimeSeries,
        params: Optional[Dict] = None,
    ) -> List[BacktestResult]:
        """
        Run the full walk-forward backtest.

        Returns a list of BacktestResult, one per fold.
        """
        params = params or {}
        y_al, x_al = y.align(x)
        folds = list(self._generate_folds(y_al, x_al))

        if not folds:
            raise ValueError(
                f"Not enough data for even one fold. "
                f"Need {self.train_size + self.test_size} bars, "
                f"got {len(y_al)}."
            )

        results = []
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            result = self._run_fold(
                fold_id, train_idx, test_idx, y_al, x_al, params
            )
            results.append(result)

        return results

    def optimize_params(
        self,
        y: PriceTimeSeries,
        x: PriceTimeSeries,
        param_grid: Dict[str, List[Any]],
        metric: str = "sharpe",
    ) -> pd.DataFrame:
        """
        Grid search over parameter combinations.

        Each combination is evaluated via a full walk-forward run.
        Results are aggregated (mean across folds) and returned as a
        sorted DataFrame.

        Parameters
        ----------
        param_grid : e.g. {"window": [30, 60, 90], "entry_threshold": [1.5, 2.0, 2.5]}
        metric     : Metric to sort by (from RiskMetrics.summary())
        """
        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))

        rows = []
        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                estimator, signal_gen = self._apply_params(params)
                engine = WalkForwardEngine(
                    estimator=estimator,
                    signal_gen=signal_gen,
                    sizer=self.sizer,
                    costs=self.costs,
                    initial_capital=self.initial_capital,
                    train_size=self.train_size,
                    test_size=self.test_size,
                    step_size=self.step_size,
                    periods_per_year=self.periods_per_year,
                )
                results = engine.run(y, x, params=params)
                agg = self._aggregate_metrics(results)
                row = {**params, **agg.to_dict(), "n_folds": len(results)}
                rows.append(row)
            except Exception as exc:
                warnings.warn(f"Params {params} failed: {exc}")

        df = pd.DataFrame(rows)
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False).reset_index(drop=True)
        return df

    def aggregate_results(self, results: List[BacktestResult]) -> pd.Series:
        """Aggregate metrics across all folds (mean)."""
        return self._aggregate_metrics(results)

    def combined_equity_curve(self, results: List[BacktestResult]) -> pd.Series:
        """Stitch together OOS equity curves across all folds."""
        curves = []
        for res in results:
            eq = res.equity_curve["equity"]
            if len(curves) > 0:
                # Rescale to continue from last equity value
                prev_eq = curves[-1].iloc[-1]
                scale = prev_eq / eq.iloc[0] if eq.iloc[0] != 0 else 1.0
                eq = eq * scale
            curves.append(eq)
        return pd.concat(curves).sort_index()

    # ------------------------------------------------------------------
    # Single fold
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        fold_id: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        y_al: PriceTimeSeries,
        x_al: PriceTimeSeries,
        params: Dict,
    ) -> BacktestResult:

        # --- Slice train/test periods ---
        train_dates = y_al.index[train_idx]
        test_dates  = y_al.index[test_idx]

        y_train = y_al.slice(train_dates[0], train_dates[-1])
        x_train = x_al.slice(train_dates[0], train_dates[-1])
        y_test  = y_al.slice(test_dates[0],  test_dates[-1])
        x_test  = x_al.slice(test_dates[0],  test_dates[-1])

        # --- Fit hedge ratio estimator on train ---
        estimator = copy.deepcopy(self.estimator)
        beta_train = estimator.fit(y_train, x_train)

        # For test period: re-run estimator on full train+test for Kalman continuity,
        # then slice to test window (preserves filter state).
        y_full = y_al.slice(train_dates[0], test_dates[-1])
        x_full = x_al.slice(train_dates[0], test_dates[-1])
        beta_full = estimator.fit(y_full, x_full)
        beta_test = beta_full.loc[test_dates[0]:test_dates[-1]]

        spread_test = SpreadSeries(
            y_test, x_test, beta_test,
            estimator_name=estimator.__class__.__name__,
        )

        # Also build train spread for signal calibration
        spread_train = SpreadSeries(
            y_train, x_train, beta_train,
            estimator_name=estimator.__class__.__name__,
        )

        # --- Fit signal generator on train spread ---
        signal_gen = copy.deepcopy(self.signal_gen)
        signal_gen.fit(spread_train)

        # --- Generate OOS signals ---
        signals = signal_gen.generate(spread_test)

        # --- Portfolio loop ---
        portfolio = Portfolio(self.initial_capital, self.costs)
        prev_direction = Direction.FLAT

        for i, (ts, sig) in enumerate(signals.items()):
            price_y = float(y_test.series.iloc[i])
            price_x = float(x_test.series.iloc[i])
            hedge_ratio = float(beta_test.iloc[i]) if i < len(beta_test) else 1.0

            qty_y, qty_x = self.sizer.size(sig, price_y, price_x, hedge_ratio)

            if sig.direction != prev_direction:
                portfolio.fill(ts, sig, qty_y, qty_x, price_y, price_x)
                prev_direction = sig.direction

            portfolio.mark(ts, price_y, price_x)

        # Close any remaining positions at end of test
        last_ts   = test_dates[-1]
        last_py   = float(y_test.series.iloc[-1])
        last_px   = float(x_test.series.iloc[-1])
        flat_sig  = Signal(Direction.FLAT, 0.0, last_ts)
        portfolio.fill(last_ts, flat_sig, 0.0, 0.0, last_py, last_px)
        portfolio.mark(last_ts, last_py, last_px)

        equity_curve = portfolio.equity_curve()
        metrics = RiskMetrics(
            equity_curve["equity"],
            risk_free_rate=0.0,
        ).summary(self.periods_per_year, trades=portfolio.trades)

        return BacktestResult(
            fold_id=fold_id,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
            equity_curve=equity_curve,
            trades=portfolio.trades,
            signals=signals,
            spread=spread_test,
            metrics=metrics,
            params=params,
        )

    # ------------------------------------------------------------------
    # Fold generation
    # ------------------------------------------------------------------

    def _generate_folds(
        self,
        y: PriceTimeSeries,
        x: PriceTimeSeries,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) for each fold."""
        n = len(y)
        start = 0
        while True:
            train_end = start + self.train_size
            test_end  = train_end + self.test_size
            if test_end > n:
                break
            train_idx = np.arange(start, train_end)
            test_idx  = np.arange(train_end, test_end)
            yield train_idx, test_idx
            start += self.step_size

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _aggregate_metrics(self, results: List[BacktestResult]) -> pd.Series:
        """Mean of each metric across folds, plus std."""
        if not results:
            return pd.Series(dtype=float)
        df = pd.DataFrame([r.metrics for r in results])
        agg = df.mean().rename(lambda c: c)
        agg_std = df.std().rename(lambda c: f"{c}_std")
        return pd.concat([agg, agg_std]).sort_index()

    def _apply_params(
        self, params: Dict
    ) -> Tuple[HedgeRatioEstimator, SignalGenerator]:
        """
        Apply a parameter dict to clones of estimator and signal_gen.
        Supports dot-notation: 'estimator.window' or 'signal.entry_threshold'.
        """
        estimator = copy.deepcopy(self.estimator)
        signal_gen = copy.deepcopy(self.signal_gen)

        for key, val in params.items():
            parts = key.split(".")
            if parts[0] == "estimator":
                setattr(estimator, parts[1], val)
            elif parts[0] == "signal":
                setattr(signal_gen, parts[1], val)
            else:
                # Try both
                if hasattr(estimator, key):
                    setattr(estimator, key, val)
                if hasattr(signal_gen, key):
                    setattr(signal_gen, key, val)

        return estimator, signal_gen
