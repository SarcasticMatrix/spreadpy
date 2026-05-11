from typing import Tuple

from abc import ABC, abstractmethod
from spreadpy.signal.signal import Signal


class PositionSizer(ABC):
    """Abstract base class for position sizers.

    Converts a :class:`Signal` into absolute quantities ``(qty_y, qty_x)`` for
    each leg. Concrete subclasses implement different sizing models (linear
    z-score scaling, inverse volatility, Kelly criterion, etc.).

    The :meth:`fit` / :meth:`size` discipline mirrors that of
    :class:`SignalGenerator`: :meth:`fit` is called once on the combined
    train + evaluation spread to warm up any rolling estimators, then
    :meth:`size` is called bar-by-bar on the evaluation period.
    """

    def fit(self, spread: "SpreadSeries") -> "PositionSizer":  # type: ignore[name-defined]
        """Calibrate the sizer on a spread series (no-op by default).

        Subclasses that require a warm-up period (e.g. :class:`InverseVolSizer`)
        override this method to precompute rolling statistics.

        :param SpreadSeries spread: Combined train + evaluation spread, used to
            warm up any rolling estimators before the first evaluation bar.

        :returns: ``self``.
        :rtype: PositionSizer
        """
        return self

    @abstractmethod
    def size(
        self,
        signal: Signal,
        price_y: float,
        price_x: float,
        hedge_ratio: float,
        capital: float = 0.0,
    ) -> Tuple[float, float]:
        """Compute absolute quantities for each leg given a signal.

        The direction of each leg is determined by the signal:

        - LONG  spread: buy y (+qty_y), sell x (−qty_x)
        - SHORT spread: sell y (−qty_y), buy x (+qty_x)
        - FLAT: returns (0, 0)

        Quantities are always returned as non-negative values; the engine
        applies the signed direction when executing fills.

        :param Signal signal: Signal at the current bar.
        :param float price_y: Current price of the y leg.
        :param float price_x: Current price of the x leg.
        :param float hedge_ratio: Hedge ratio β_t at the current bar.
        :param float capital: Current mark-to-market equity in monetary units.
            Used by capital-fraction sizers (e.g. :class:`KellyTruncatedEntry`);
            ignored by notional-based sizers.

        :returns: ``(qty_y, qty_x)`` — absolute quantities for each leg.
        :rtype: Tuple[float, float]
        """
