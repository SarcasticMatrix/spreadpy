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

    .. math::

        p_{\\text{fill}} &= p_{\\text{mid}} \\cdot (1 + d \\cdot s / 10^4)
            && (d = +1 \\text{ for buys, } -1 \\text{ for sells}) \\\\
        c_{\\text{slip}} &= |p_{\\text{fill}} - p_{\\text{mid}}| \\cdot q \\\\
        c_{\\text{comm}} &= \\max\\!\\left(c_{\\text{unit}} \\cdot q
            + |p_{\\text{fill}} \\cdot q| \\cdot b / 10^4,\\;
            c_{\\min}\\right) \\\\
        c_{\\text{total}} &= c_{\\text{slip}} + c_{\\text{comm}}

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

        .. math::

            p_{\\text{fill}} = p \\cdot (1 + d \\cdot s / 10^4)

        The total monetary cost is:

        .. math::

            c_{\\text{slip}}  &= |p_{\\text{fill}} - p| \\cdot q \\\\
            c_{\\text{comm}}  &= \\max\\!\\left(c_{\\text{unit}} \\cdot q
                + |p_{\\text{fill}} \\cdot q| \\cdot b / 10^4,\\;
                c_{\\min}\\right) \\\\
            c_{\\text{total}} &= c_{\\text{slip}} + c_{\\text{comm}}

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