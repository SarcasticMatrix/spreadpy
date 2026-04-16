"""
backtest/costs.py — Transaction cost model
TransactionCosts
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
    commission_bps : float
        Commission as a fraction of notional (e.g. 0.0001 = 1 bps) in basis points.
    min_commission : float
        Minimum per-trade commission.

    Cost model per trade
    --------------------
    total = slippage_cost +
        max(commission_per_unit * qty + notional * commission_bps/100, min_commission)
    """

    slippage_bps: float = 2.0
    commission_per_unit: float = 0.0
    commission_bps: float = 1.0   # 1 bps
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
            self.commission_per_unit * abs(qty) + notional * self.commission_bps/100,
            self.min_commission,
        )
        total_cost = abs(slip * qty) + commission
        return fill_price, total_cost