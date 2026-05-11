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
    min_commission: float = 4.90

    def apply(self, price: float, qty: float, direction: int) -> Tuple[float, float]:
        """Compute the fill price and total cost for a single leg fill.

        The fill price incorporates one-way adverse slippage:

            fill_price = price · (1 + direction · slippage_bps / 10 000)

        The total monetary cost is:

            slippage   = |fill_price − price| · qty
            commission = max(commission_per_unit · qty
                             + |fill_price · qty| · commission_bps / 10 000,
                             min_commission)
            total_cost = slippage + commission

        :param float price: Mid price at signal time.
        :param float qty: Absolute quantity (always positive).
        :param int direction: Fill direction: +1 (buy) or −1 (sell).

        :returns: ``(fill_price, total_cost)`` where ``total_cost`` is always
            positive.
        :rtype: Tuple[float, float]
        """
        slip = price * (self.slippage_bps / 10_000) * direction
        fill_price = price + slip

        notional = abs(fill_price * qty)
        commission = max(
            self.commission_per_unit * abs(qty) + notional * self.commission_bps / 10_000,
            self.min_commission,
        )
        total_cost = abs(slip * qty) + commission
        return fill_price, total_cost