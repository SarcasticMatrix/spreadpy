from typing import Callable, Optional, Tuple

import numpy as np

from spreadpy.signal.signal import Signal, Direction

class PositionSizer:
    """
    Converts a Signal into (qty_y, qty_x) pair given available capital
    and the current hedge ratio.

    Position sizing is driven by the z-score magnitude (|z|):
        notional_y = max_notional * scale_fn(|z|)
        notional_x = notional_y  (hedge ratio adjusts qty_x)

    Parameters
    ----------
    max_notional : float
        Maximum notional per leg in monetary units.
    scale_fn     : Callable[[float], float]
        Maps |z| → [0, 1] fraction of max_notional.
        Default: linear ramp capped at 1 between z=1 and z=3.
    """

    def __init__(
        self,
        max_notional: float = 100_000.0,
        scale_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.max_notional = max_notional
        self.scale_fn = scale_fn or self._default_scale

    @staticmethod
    def _default_scale(abs_z: float) -> float:
        """Linear interpolation from 0 at z=1 to 1 at z=3."""
        return float(np.clip((abs_z - 1.0) / 2.0, 0.0, 1.0))

    def size(
        self,
        signal: Signal,
        price_y: float,
        price_x: float,
        hedge_ratio: float,
    ) -> Tuple[float, float]:
        """
        Compute absolute quantities for each leg.

        Returns
        -------
        qty_y : Quantity of asset y (direction determined by signal)
        qty_x : Quantity of asset x (direction is opposite to y, scaled by β)
        """
        if signal.direction == Direction.FLAT or np.isnan(signal.zscore):
            return 0.0, 0.0

        abs_z = abs(signal.zscore)
        scale = self.scale_fn(abs_z)
        notional_y = self.max_notional * scale

        qty_y = notional_y / price_y if price_y > 0 else 0.0
        qty_x = (notional_y * abs(hedge_ratio)) / price_x if price_x > 0 else 0.0

        return qty_y, qty_x
