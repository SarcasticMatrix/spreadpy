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

    .. math::

        \\text{Sharpe}  &= \\frac{\\mathbb{E}[r - r_f]}{\\sigma[r]} \\cdot \\sqrt{T} \\\\
        \\text{Sortino} &= \\frac{\\text{CAGR} - r_f}{\\sigma_{\\downarrow}} \\cdot \\sqrt{T}
                         \\quad (\\text{downside std below MAR}) \\\\
        \\text{Calmar}  &= \\frac{\\text{CAGR}}{|\\text{MaxDD}|} \\\\
        \\text{MaxDD}   &= \\min_t\\!\\left(\\frac{\\text{equity}_t}
                         {\\max_{s \\leq t}\\, \\text{equity}_s} - 1\\right)

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
        """Compute the total (cumulative) return over the evaluation period.

        Defined as:

        .. math::

            R = \\frac{\\text{equity}_T}{\\text{equity}_0} - 1

        :returns: Total return as a fraction (e.g. 0.12 for +12%).
        :rtype: float
        """
        if len(self.equity) < 2:
            return 0.0
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1)

    def annualised_return(self, periods_per_year: int = 252) -> float:
        """Compute the Compound Annual Growth Rate (CAGR).

        Defined as:

        .. math::

            \\text{CAGR} = (1 + R)^{T/n} - 1

        where :math:`R` is the total return, :math:`n` the number of bars,
        and :math:`T` = ``periods_per_year``.

        :param int periods_per_year: Number of bars per year (252 for daily).

        :returns: Annualised return as a fraction. Returns −1 if total equity
            is non-positive.
        :rtype: float
        """
        n = len(self._returns)
        if n < 2:
            return 0.0
        total = self.total_return()
        if 1 + total <= 0:
            return -1.0          # total wipeout or worse — CAGR undefined
        return float((1 + total) ** (periods_per_year / n) - 1)

    def volatility(self, periods_per_year: int = 252) -> float:
        """Compute the annualised volatility of bar-level returns.

        Defined as :math:`\\sigma_{\\text{ann}} = \\sigma_{\\text{bar}} \\cdot \\sqrt{T}`
        where :math:`\\sigma_{\\text{bar}}` is the sample standard deviation of
        simple returns and :math:`T` = ``periods_per_year``.

        :param int periods_per_year: Number of bars per year (252 for daily).

        :returns: Annualised volatility as a fraction.
        :rtype: float
        """
        return float(self._returns.std() * np.sqrt(periods_per_year))

    def sharpe(self, periods_per_year: int = 252) -> float:
        """Compute the annualised Sharpe ratio.

        Defined as:

        .. math::

            \\text{Sharpe} = \\frac{\\mathbb{E}[r_t - r_f/T]}{\\sigma_{\\text{bar}}} \\cdot \\sqrt{T}

        where :math:`r_t` is the bar-level return, :math:`T` = ``periods_per_year``,
        and :math:`\\sigma_{\\text{bar}}` the sample standard deviation of bar returns.

        :param int periods_per_year: Number of bars per year (252 for daily).

        :returns: Annualised Sharpe ratio. Returns 0 if volatility is zero.
        :rtype: float
        """
        vol = self.volatility(periods_per_year)
        if vol == 0:
            return 0.0
        excess = self._returns - self.rfr / periods_per_year
        return float(excess.mean() / self._returns.std() * np.sqrt(periods_per_year))

    def sortino(self, periods_per_year: int = 252, mar: float = 0.0) -> float:
        """Compute the annualised Sortino ratio.

        Uses downside deviation below the minimum acceptable return (MAR)
        as the risk denominator:

        .. math::

            \\sigma_{\\downarrow} &= \\sqrt{\\mathbb{E}[\\min(r_t - \\text{mar},\\, 0)^2]}
                                  \\cdot \\sqrt{T} \\\\
            \\text{Sortino} &= \\frac{\\text{CAGR} - r_f}{\\sigma_{\\downarrow}}

        :param int periods_per_year: Number of bars per year (252 for daily).
        :param float mar: Minimum acceptable return per bar (default 0.0).

        :returns: Annualised Sortino ratio. Returns ``+inf`` if there are no
            returns below ``mar``.
        :rtype: float
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
        """Compute the maximum peak-to-trough drawdown.

        Defined as:

        .. math::

            \\text{MDD} = \\min_t\\!\\left(
                \\frac{\\text{equity}_t}{\\max_{s \\leq t}\\, \\text{equity}_s} - 1
            \\right)

        :returns: Maximum drawdown as a negative fraction (e.g. −0.15 for −15%).
        :rtype: float
        """
        roll_max = self.equity.cummax()
        drawdown = (self.equity - roll_max) / roll_max
        return float(drawdown.min())

    def calmar(self, periods_per_year: int = 252) -> float:
        """Compute the Calmar ratio.

        Defined as:

        .. math::

            \\text{Calmar} = \\frac{\\text{CAGR}}{|\\text{MDD}|}

        :param int periods_per_year: Number of bars per year (252 for daily).

        :returns: Calmar ratio. Returns ``+inf`` if max drawdown is zero.
        :rtype: float
        """
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return float("inf")
        return float(self.annualised_return(periods_per_year) / mdd)

    def drawdown_series(self) -> pd.Series:
        """Compute the full drawdown time series.

        At each bar :math:`t`:

        .. math::

            DD_t = \\frac{\\text{equity}_t}{\\max_{s \\leq t}\\, \\text{equity}_s} - 1

        :returns: Drawdown series (values ≤ 0) aligned with ``self.equity.index``.
        :rtype: pd.Series
        """
        roll_max = self.equity.cummax()
        return (self.equity - roll_max) / roll_max

    def avg_drawdown(self) -> float:
        """Compute the average drawdown over all sub-zero drawdown bars.

        :returns: Mean drawdown as a negative fraction, or 0 if no bar is
            in drawdown.
        :rtype: float
        """
        dd = self.drawdown_series()
        below = dd[dd < 0]
        return float(below.mean()) if len(below) > 0 else 0.0

    def conditional_drawdown(self, alpha: float = 0.05) -> float:
        """
        Conditional Drawdown at Risk (CDaR) at level alpha.

        Average of the worst alpha-fraction of drawdown observations:

        .. math::

            \\text{CDaR}_\\alpha = \\mathbb{E}[DD_t \\mid DD_t \\leq \\text{VaR}_\\alpha(DD)]

        where :math:`\\text{VaR}_\\alpha` is the :math:`\\alpha`-quantile of the
        drawdown distribution. Returns a negative value (same sign convention
        as :meth:`max_drawdown`).

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
        """Compute the fraction of round-trip trades with positive P&L.

        Round trips are matched FIFO per leg via :meth:`_compute_round_trips`.

        :param List[Trade] trades: List of leg fills from :attr:`Portfolio.trades`.

        :returns: Win rate ∈ [0, 1], or NaN if no round trips exist.
        :rtype: float
        """
        round_trips = self._compute_round_trips(trades)
        if not round_trips:
            return float("nan")
        wins = sum(1 for pnl in round_trips if pnl > 0)
        return wins / len(round_trips)

    def profit_factor(self, trades: List["Trade"]) -> float:
        """Compute the profit factor (gross profits divided by gross losses).

        Defined as:

        .. math::

            \\text{PF} = \\frac{\\sum_{\\text{winning}} \\text{P\\&L}}
                               {\\left|\\sum_{\\text{losing}} \\text{P\\&L}\\right|}

        :param List[Trade] trades: List of leg fills from :attr:`Portfolio.trades`.

        :returns: Profit factor > 1 indicates net profitability. Returns
            ``+inf`` if there are no losing trades.
        :rtype: float
        """
        round_trips = self._compute_round_trips(trades)
        gains = sum(pnl for pnl in round_trips if pnl > 0)
        losses = abs(sum(pnl for pnl in round_trips if pnl < 0))
        return gains / losses if losses > 0 else float("inf")

    def turnover(self, trades: List["Trade"], periods_per_year: int = 252) -> float:
        """Compute annualised portfolio turnover.

        Defined as:

        .. math::

            \\text{turnover} = \\frac{\\sum_i |N_i| \\cdot T}{n \\cdot \\bar{\\text{equity}}}

        where :math:`n` is the number of bars in the equity series,
        :math:`T` = ``periods_per_year``, and :math:`\\bar{\\text{equity}}` is
        the mean equity over the period.

        :param List[Trade] trades: List of leg fills from :attr:`Portfolio.trades`.
        :param int periods_per_year: Number of bars per year (252 for daily).

        :returns: Annualised turnover as a multiple of average equity.
        :rtype: float
        """
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
        """Return all key risk metrics as a labelled series.

        Always includes: ``total_return``, ``annualised_return``,
        ``volatility``, ``sharpe``, ``sortino``, ``max_drawdown``,
        ``avg_drawdown``, ``cdar_5``, ``calmar``.

        When ``trades`` is provided, also includes: ``win_rate``,
        ``profit_factor``, ``n_trades``.

        :param int periods_per_year: Number of bars per year (252 for daily).
        :param Optional[List[Trade]] trades: Leg fills used to compute
            trade-level metrics.

        :returns: Named series of risk metrics.
        :rtype: pd.Series
        """
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
        """Pair entry and exit fills per leg to compute round-trip P&L (FIFO).

        For each leg (y and x), buys (+1) are pushed onto a stack and matched
        FIFO against sells (−1). The P&L of each matched pair is:

        .. math::

            \\text{pnl} = (\\text{exit\\_fill\\_price} - \\text{entry\\_fill\\_price}) \\cdot \\text{qty}

        :param List[Trade] trades: List of leg fills from :attr:`Portfolio.trades`.

        :returns: List of round-trip P&L values in monetary units.
        :rtype: List[float]
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
