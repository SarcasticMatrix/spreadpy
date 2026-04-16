"""
backtest/metrics.py — Risk metrics
RiskMetrics
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from spreadpy.backtest.portfolio import Trade


class RiskMetrics:
    """
    Stateless risk metric calculator for a backtest equity curve.

    All methods operate on the equity series supplied at construction
    and return scalar values. Annualisation uses the ``periods_per_year``
    argument passed to each method.

    Key metrics:

        Sharpe  = E[r − r_f] / σ[r] · √T
        Sortino = (CAGR − r_f) / σ_down · √T     (downside std below MAR)
        Calmar  = CAGR / |MaxDD|
        MaxDD   = min_t  (equity_t / max_{s≤t} equity_s  − 1)

    :param pd.Series equity: Bar-level or daily equity curve (NaNs are dropped).
    :param float risk_free_rate: Annual risk-free rate r_f (e.g. 0.04 for 4%).
    """

    def __init__(self, equity: pd.Series, risk_free_rate: float = 0.0) -> None:
        if isinstance(equity, pd.DataFrame):
            equity = equity["equity"]
        self.equity = equity.dropna()
        self.rfr = risk_free_rate
        self._returns: pd.Series = self.equity.pct_change().dropna()

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def total_return(self) -> float:
        """Total return over the period."""
        if len(self.equity) < 2:
            return 0.0
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1)

    def annualised_return(self, periods_per_year: int = 252) -> float:
        """CAGR."""
        n = len(self._returns)
        if n < 2:
            return 0.0
        total = self.total_return()
        if 1 + total <= 0:
            return -1.0          # total wipeout or worse — CAGR undefined
        return float((1 + total) ** (periods_per_year / n) - 1)

    def volatility(self, periods_per_year: int = 252) -> float:
        """Annualised volatility of returns."""
        return float(self._returns.std() * np.sqrt(periods_per_year))

    def sharpe(self, periods_per_year: int = 252) -> float:
        """Annualised Sharpe ratio."""
        vol = self.volatility(periods_per_year)
        if vol == 0:
            return 0.0
        excess = self._returns - self.rfr / periods_per_year
        return float(excess.mean() / self._returns.std() * np.sqrt(periods_per_year))

    def sortino(self, periods_per_year: int = 252, mar: float = 0.0) -> float:
        """
        Sortino ratio using downside deviation below MAR.

        Parameters
        ----------
        mar : Minimum acceptable return per period (default 0)
        """
        downside = self._returns[self._returns < mar]
        if len(downside) == 0:
            return float("inf")
        downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year)
        if downside_std == 0:
            return float("inf")
        excess_return = self.annualised_return(periods_per_year) - self.rfr
        return float(excess_return / downside_std)

    def max_drawdown(self) -> float:
        """Maximum drawdown (as a negative fraction)."""
        roll_max = self.equity.cummax()
        drawdown = (self.equity - roll_max) / roll_max
        return float(drawdown.min())

    def calmar(self, periods_per_year: int = 252) -> float:
        """Calmar ratio = annualised return / |max drawdown|."""
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return float("inf")
        return float(self.annualised_return(periods_per_year) / mdd)

    def drawdown_series(self) -> pd.Series:
        """Full drawdown time series (fraction, always <= 0)."""
        roll_max = self.equity.cummax()
        return (self.equity - roll_max) / roll_max

    def avg_drawdown(self) -> float:
        """Average drawdown (mean of all sub-zero drawdown values)."""
        dd = self.drawdown_series()
        below = dd[dd < 0]
        return float(below.mean()) if len(below) > 0 else 0.0

    def conditional_drawdown(self, alpha: float = 0.05) -> float:
        """
        Conditional Drawdown at Risk (CDaR) at level alpha.

        Average of the worst alpha-fraction of drawdown observations:

            CDaR_α = E[DD_t | DD_t ≤ VaR_α(DD)]

        where VaR_α is the alpha-quantile of the drawdown distribution.
        Returns a negative value (same sign convention as max_drawdown).

        :param float alpha: Tail level (default 0.05 = worst 5%).
        :returns: Mean drawdown in the worst alpha fraction of bars.
        :rtype: float
        """
        dd = self.drawdown_series()
        if len(dd) == 0:
            return 0.0
        threshold = dd.quantile(alpha)          # α-quantile (≤ 0)
        tail = dd[dd <= threshold]
        return float(tail.mean()) if len(tail) > 0 else 0.0

    def win_rate(self, trades: List["Trade"]) -> float:
        """Fraction of round-trip trades with positive P&L."""
        round_trips = self._compute_round_trips(trades)
        if not round_trips:
            return float("nan")
        wins = sum(1 for pnl in round_trips if pnl > 0)
        return wins / len(round_trips)

    def profit_factor(self, trades: List["Trade"]) -> float:
        """Gross profits / gross losses."""
        round_trips = self._compute_round_trips(trades)
        gains = sum(pnl for pnl in round_trips if pnl > 0)
        losses = abs(sum(pnl for pnl in round_trips if pnl < 0))
        return gains / losses if losses > 0 else float("inf")

    def turnover(self, trades: List["Trade"], periods_per_year: int = 252) -> float:
        """Annualised turnover = total traded notional / average equity."""
        if not trades:
            return 0.0
        total_notional = sum(t.notional for t in trades)
        avg_equity = self.equity.mean()
        if avg_equity == 0:
            return float("inf")
        return float(total_notional * periods_per_year / len(self.equity) / avg_equity)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(
        self,
        periods_per_year: int = 252,
        trades: Optional[List] = None,
    ) -> pd.Series:
        """Return a pd.Series with all key metrics."""
        data = {
            "total_return":       self.total_return(),
            "annualised_return":  self.annualised_return(periods_per_year),
            "volatility":         self.volatility(periods_per_year),
            "sharpe":             self.sharpe(periods_per_year),
            "sortino":            self.sortino(periods_per_year),
            "max_drawdown":       self.max_drawdown(),
            "avg_drawdown":       self.avg_drawdown(),
            "cdar_5":             self.conditional_drawdown(alpha=0.05),
            "calmar":             self.calmar(periods_per_year),
        }
        if trades is not None:
            data["win_rate"]      = self.win_rate(trades)
            data["profit_factor"] = self.profit_factor(trades)
            data["n_trades"]      = len(self._compute_round_trips(trades)) // 2
        return pd.Series(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_round_trips(trades: List["Trade"]) -> List[float]:
        """
        Pair entry and exit trades per leg to compute round-trip P&L.
        FIFO matching per leg.
        """
        pnls = []
        for leg in ("y", "x"):
            leg_trades = [t for t in trades if t.leg == leg]
            stack = []
            for t in leg_trades:
                if t.direction == +1:
                    stack.append(t.fill_price)
                elif t.direction == -1 and stack:
                    entry = stack.pop(0)
                    pnls.append((t.fill_price - entry) * t.qty)
        return pnls
