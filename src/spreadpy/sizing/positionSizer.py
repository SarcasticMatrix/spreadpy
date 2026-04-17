from typing import Tuple

from abc import ABC, abstractmethod
from spreadpy.signal.signal import Signal


class PositionSizer(ABC):
    """
    Converts a Signal into (qty_y, qty_x) absolute quantities.
    """

    @abstractmethod
    def size(
        self,
        signal: Signal,
        price_y: float,
        price_x: float,
        hedge_ratio: float,
        capital: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Compute absolute quantities for each leg.

        The ``capital`` parameter is accepted for interface compatibility with
        :class:`KellySizer` but is not used here.

        Returns
        -------
        qty_y : Quantity of asset y (direction determined by signal)
        qty_x : Quantity of asset x (direction is opposite to y, scaled by β)
        """
