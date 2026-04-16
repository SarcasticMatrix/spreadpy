"""
backtest/portfolio.py — Portfolio state management
Trade, Portfolio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from spreadpy.signal import Direction, Signal
from spreadpy.backtest.costs import TransactionCosts


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
    Call fill() at each signal bar to update positions.
    Call mark() at each bar to record equity.
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
        self._equity_records: List[Dict] = []
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
            direction = -1 if qty > 0 else +1
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

        if old_qty != 0 and np.sign(old_qty) != np.sign(signed_qty):
            closing_qty = min(abs(old_qty), abs(signed_qty))
            realised = closing_qty * (trade.fill_price - old_price) * np.sign(old_qty)
            self._realised_pnl += realised
            self._cash += realised

        new_qty = old_qty + signed_qty
        if abs(new_qty) < 1e-9:
            pos["qty"] = 0.0
            pos["avg_price"] = 0.0
        elif np.sign(new_qty) == np.sign(old_qty):
            pos["avg_price"] = (
                (abs(old_qty) * old_price + trade.qty * trade.fill_price)
                / abs(new_qty)
            )
            pos["qty"] = new_qty
        else:
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
