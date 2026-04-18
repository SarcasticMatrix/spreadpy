from typing import Callable, Optional, Tuple

import numpy as np
from scipy.stats import norm

from spreadpy.signal.signal import Signal, Direction
from spreadpy.sizing import PositionSizer


class LinearSizer(PositionSizer):
    """
    Converts a Signal into (qty_y, qty_x) absolute quantities.

    The position size is proportional to the z-score magnitude via a scaling function:

        notional_y = max_notional · scale(|z|)
        qty_y      = notional_y / price_y
        qty_x      = notional_y · |β| / price_x

    where β is the hedge ratio at signal time. The x-leg quantity is adjusted
    so that the notional exposure is hedged: qty_x · price_x ≈ |β| · qty_y · price_y.

    The default scale function is a linear ramp:

        scale(|z|) = clip((|z| − 1) / 2,  0,  1)

    mapping |z| = 1 → 0% and |z| ≥ 3 → 100% of ``max_notional``.

    :param float max_notional: Maximum notional per leg in monetary units.
    :param Optional[Callable[[float], float]] scale_fn: Maps |z| → [0, 1].
        Defaults to the linear ramp described above.
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
        capital: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Compute absolute quantities for each leg.

        The ``capital`` parameter is accepted for interface compatibility with
        :class:`KellyTruncatedEntry` but is not used here.

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
