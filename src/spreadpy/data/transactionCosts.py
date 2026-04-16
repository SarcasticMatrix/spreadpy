"""
data.py — Layer 1: Data primitives
PriceTimeSeries, DataLoader, TransactionCosts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple



@dataclass
class TransactionCosts:
    """
    Models transaction costs applied at fill time.

    Parameters
    ----------
    slippage_bps : float
        One-way slippage in basis points (applied as adverse price move).
    commission_per_unit : float
        Fixed commission per unit traded (in price currency).
    commission_pct : float
        Commission as a fraction of notional (e.g. 0.0001 = 1 bps).
    min_commission : float
        Minimum per-trade commission.

    Cost model per trade
    --------------------
    total = slippage_cost + 
        max(commission_per_unit * qty + notional * commission_pct, min_commission)
    """

    slippage_bps: float = 2.0
    commission_per_unit: float = 0.0
    commission_pct: float = 0.0001   # 1 bps
    min_commission: float = 0.0

    def apply(self, price: float, qty: float, direction: int) -> Tuple[float, float]:
        """
        Compute fill price and cost for a single leg.

        Parameters
        ----------
        price     : Mid price at signal time
        qty       : Absolute quantity
        direction : +1 (buy) or -1 (sell)

        Returns
        -------
        fill_price : Price after slippage
        cost       : Total monetary cost (always positive)
        """
        slip = price * (self.slippage_bps / 10_000) * direction
        fill_price = price + slip

        notional = abs(fill_price * qty)
        commission = max(
            self.commission_per_unit * abs(qty) + notional * self.commission_pct,
            self.min_commission,
        )
        total_cost = abs(slip * qty) + commission
        return fill_price, total_cost

    def round_trip_cost_bps(self, price: float) -> float:
        """Approximate round-trip cost in bps (useful for quick sanity checks)."""
        _, entry = self.apply(price, 1.0, +1)
        _, exit_ = self.apply(price, 1.0, -1)
        return (entry + exit_) / price * 10_000
