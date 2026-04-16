"""
sizing.py — Layer 4: Position sizing, portfolio, risk metrics
PositionSizer, Trade, Portfolio, RiskMetrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from spreadpy.signal import Direction, Signal
from spreadpy.data import TransactionCosts


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """
    Represents a single fill on one leg of the spread.

    Attributes
    ----------
    timestamp       : Fill time
    leg             : 'y' or 'x'
    direction       : +1 (buy) or -1 (sell)
    qty             : Absolute quantity
    price           : Mid price at signal
    fill_price      : Price after slippage
    cost            : Total transaction cost (monetary)
    signal          : The Signal that triggered this trade
    """

    timestamp: pd.Timestamp
    leg: str
    direction: int
    qty: float
    price: float
    fill_price: float
    cost: float
    signal: Optional[Signal] = None

    @property
    def pnl_contribution(self) -> float:
        """Unrealised P&L from fill_price to mark price (requires mark later)."""
        return 0.0  # updated externally by Portfolio

    @property
    def notional(self) -> float:
        return self.fill_price * self.qty



# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class Portfolio:
    """
    Maintains the state of the backtest portfolio.

    Tracks:
    - Open positions per leg (qty and avg_fill_price)
    - Realised P&L and accrued costs
    - Equity curve (mark-to-market)

    Usage
    -----
    Call fill(trade, mark_price_y, mark_price_x) at each bar to update state.
    Call mark(price_y, price_x) at each bar to update unrealised P&L.
    Call equity_curve() at the end to retrieve the full equity series.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        costs: Optional[TransactionCosts] = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.costs = costs or TransactionCosts()

        self._cash: float = initial_capital
        self._realised_pnl: float = 0.0
        self._total_costs: float = 0.0

        # Position state: leg → {qty (signed), avg_fill_price}
        self._positions: Dict[str, Dict] = {
            "y": {"qty": 0.0, "avg_price": 0.0},
            "x": {"qty": 0.0, "avg_price": 0.0},
        }

        self._trades: List[Trade] = []
        self._equity_records: List[Dict] = []  # {timestamp, equity, cash, unrealised}
        self._current_prices: Dict[str, float] = {"y": 0.0, "x": 0.0}

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def fill(
        self,
        timestamp: pd.Timestamp,
        signal: Signal,
        qty_y: float,
        qty_x: float,
        price_y: float,
        price_x: float,
    ) -> List[Trade]:
        if signal.direction == Direction.FLAT:
            return self._close_all(timestamp, signal, price_y, price_x)

        trades = []
        existing_qty = self._positions["y"]["qty"]
        if abs(existing_qty) > 1e-9:
            close_sig = Signal(Direction.FLAT, signal.zscore, timestamp)
            trades.extend(self._close_all(timestamp, close_sig, price_y, price_x))

        # Ouvre la nouvelle position
        dir_y = int(signal.direction)
        dir_x = -dir_y

        for leg, direction, qty, price in [
            ("y", dir_y,  qty_y, price_y),
            ("x", dir_x,  qty_x, price_x),
        ]:
            if qty <= 0:
                continue
            fill_price, cost = self.costs.apply(price, qty, direction)
            trade = Trade(
                timestamp=timestamp,
                leg=leg,
                direction=direction,
                qty=qty,
                price=price,
                fill_price=fill_price,
                cost=cost,
                signal=signal,
            )
            self._apply_trade(leg, trade)
            trades.append(trade)

        self._trades.extend(trades)
        self._current_prices = {"y": price_y, "x": price_x}
        return trades

    def _close_all(
        self,
        timestamp: pd.Timestamp,
        signal: Signal,
        price_y: float,
        price_x: float,
    ) -> List[Trade]:
        """Close all open positions (flat signal)."""
        trades = []
        for leg, price in [("y", price_y), ("x", price_x)]:
            qty = self._positions[leg]["qty"]
            if abs(qty) < 1e-9:
                continue
            direction = -1 if qty > 0 else +1  # close direction
            fill_price, cost = self.costs.apply(price, abs(qty), direction)
            trade = Trade(
                timestamp=timestamp,
                leg=leg,
                direction=direction,
                qty=abs(qty),
                price=price,
                fill_price=fill_price,
                cost=cost,
                signal=signal,
            )
            self._apply_trade(leg, trade)
            trades.append(trade)
        self._trades.extend(trades)
        self._current_prices = {"y": price_y, "x": price_x}
        return trades

    def _apply_trade(self, leg: str, trade: Trade) -> None:
        """Update position and realised P&L for a single leg."""
        pos = self._positions[leg]
        old_qty = pos["qty"]
        old_price = pos["avg_price"]
        signed_qty = trade.direction * trade.qty

        # Realised P&L on the closing portion
        if old_qty != 0 and np.sign(old_qty) != np.sign(signed_qty):
            closing_qty = min(abs(old_qty), abs(signed_qty))
            realised = closing_qty * (trade.fill_price - old_price) * np.sign(old_qty)
            self._realised_pnl += realised
            self._cash += realised

        # Update position (average price tracking)
        new_qty = old_qty + signed_qty
        if abs(new_qty) < 1e-9:
            pos["qty"] = 0.0
            pos["avg_price"] = 0.0
        elif np.sign(new_qty) == np.sign(old_qty):
            # Adding to position: weighted average price
            pos["avg_price"] = (
                (abs(old_qty) * old_price + trade.qty * trade.fill_price)
                / abs(new_qty)
            )
            pos["qty"] = new_qty
        else:
            # Flipped position
            pos["qty"] = new_qty
            pos["avg_price"] = trade.fill_price

        self._cash -= trade.cost
        self._total_costs += trade.cost

    def mark(self, timestamp: pd.Timestamp, price_y: float, price_x: float) -> float:
        """
        Mark-to-market: compute current equity and record it.
        Should be called at every bar (even when no trade occurs).
        """
        unrealised = 0.0
        for leg, price in [("y", price_y), ("x", price_x)]:
            pos = self._positions[leg]
            if abs(pos["qty"]) > 1e-9:
                unrealised += pos["qty"] * (price - pos["avg_price"])

        equity = self._cash + unrealised
        self._equity_records.append({
            "timestamp": timestamp,
            "equity": equity,
            "cash": self._cash,
            "unrealised_pnl": unrealised,
            "realised_pnl": self._realised_pnl,
            "total_costs": self._total_costs,
        })
        self._current_prices = {"y": price_y, "x": price_x}
        return equity

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def equity_curve(self) -> pd.DataFrame:
        """Return the full equity curve as a DataFrame."""
        df = pd.DataFrame(self._equity_records).set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        return df

    @property
    def trades(self) -> List[Trade]:
        return self._trades

    @property
    def total_costs(self) -> float:
        return self._total_costs

    @property
    def positions(self) -> Dict[str, Dict]:
        return self._positions

    def reset(self) -> None:
        """Reset portfolio to initial state (used between walk-forward folds)."""
        self.__init__(self.initial_capital, self.costs)


# ---------------------------------------------------------------------------
# RiskMetrics
# ---------------------------------------------------------------------------

class RiskMetrics:
    """
    Stateless risk metric calculator.
    All methods take an equity curve (pd.Series or equity column of DataFrame).
    """

    def __init__(self, equity: pd.Series, risk_free_rate: float = 0.0) -> None:
        """
        Parameters
        ----------
        equity         : pd.Series of equity values (daily or bar-level)
        risk_free_rate : Annual risk-free rate (e.g. 0.04 for 4%)
        """
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
            "calmar":             self.calmar(periods_per_year),
        }
        if trades is not None:
            data["win_rate"]      = self.win_rate(trades)
            data["profit_factor"] = self.profit_factor(trades)
            data["n_trades"]      = len(trades) // 2  # round trips (2 legs)
        return pd.Series(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_round_trips(trades: List["Trade"]) -> List[float]:
        """
        Pair entry and exit trades per leg to compute round-trip P&L.
        Very simplified: FIFO matching per leg.
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
