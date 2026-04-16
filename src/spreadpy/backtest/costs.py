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
    Transaction cost model applied at each leg fill.

    The total cost per fill is computed as:

        fill_price  = mid ± (slippage_bps / 10 000) · mid    (+ for buys, − for sells)
        slippage    = |fill_price − mid| · qty
        commission  = max(commission_per_unit · qty
                          + notional · commission_bps / 10 000,
                          min_commission)
        total_cost  = slippage + commission

    :param float slippage_bps: One-way adverse price move in basis points.
    :param float commission_per_unit: Fixed commission charged per unit traded.
    :param float commission_bps: Ad-valorem commission on notional, in basis points
        (e.g. 1.0 = 1 bps = 0.01%).
    :param float min_commission: Minimum commission floor per fill.
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