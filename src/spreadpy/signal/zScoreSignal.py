from typing import Optional

import numpy as np
import pandas as pd

from spreadpy.signal.signal import SignalGenerator, Signal, Direction
from spreadpy.spread.spreadSeries import SpreadSeries


class ZScoreSignal(SignalGenerator):
    """
    Classic z-score entry / exit signal generator.

    At each bar, computes a rolling z-score of the spread residuals:

        z_t = (s_t − μ̂_t) / σ̂_t

    where μ̂_t and σ̂_t are the rolling mean and standard deviation over
    the last ``window`` bars (no lookahead).

    Entry / exit rules:

        LONG   if z_t < −entry_threshold
        SHORT  if z_t > +entry_threshold
        FLAT   if LONG  and z_t > −revert_threshold  (z reverted back up)
        FLAT   if SHORT and z_t < +revert_threshold  (z reverted back down)

    Setting ``revert_threshold=0`` exits at the mean crossing (z crosses 0).
    Setting it to a positive value exits before the mean is fully reached.

    :param int window: Number of bars for the rolling z-score computation.
    :param float entry_threshold: |z| level above which a position is opened.
    :param float revert_threshold: z level at which mean reversion is considered
        complete and the position is closed. Must be ≤ entry_threshold.
        Use 0.0 to exit at the mean (default).
    """

    def __init__(
        self,
        window: int = 60,
        entry_threshold: float = 1.0,
        revert_threshold: float = 0.0,
    ) -> None:
        self.window = window
        self.entry_threshold = entry_threshold
        self.revert_threshold = revert_threshold

        # Fitted attributes
        self._mu: Optional[float] = None
        self._sigma: Optional[float] = None

    def fit(self, spread: SpreadSeries) -> "ZScoreSignal":
        """Compute in-sample mean and std for normalisation."""
        residuals = spread.residuals.dropna()
        self._mu = float(residuals.mean())
        self._sigma = float(residuals.std())
        return self

    def generate(self, spread: SpreadSeries) -> pd.Series:
        """
        Compute rolling z-score and map to Signal objects.
        Uses rolling statistics (no lookahead) regardless of fit().
        fit() stats can optionally be used as a fallback.
        """
        residuals = spread.residuals

        # Rolling z-score (no lookahead)
        roll_mu = residuals.rolling(self.window).mean()
        roll_sigma = residuals.rolling(self.window).std().replace(0, np.nan)
        zscore = (residuals - roll_mu) / roll_sigma

        signals = []
        prev_direction = Direction.FLAT

        for ts, z in zscore.items():
            if np.isnan(z):
                signals.append(Signal(Direction.FLAT, float("nan"), ts))
                continue

            # --- Exit condition (checked first) ---
            if prev_direction != Direction.FLAT:
                reverted = (
                    (prev_direction == Direction.LONG  and z > -self.revert_threshold) or
                    (prev_direction == Direction.SHORT and z <  self.revert_threshold)
                )
                if reverted:
                    sig = Signal(Direction.FLAT, z, ts, is_entry=False)
                    prev_direction = Direction.FLAT
                    signals.append(sig)
                    continue

            # --- Entry conditions ---
            if z < -self.entry_threshold:
                direction = Direction.LONG
                is_entry = prev_direction != Direction.LONG
            elif z > self.entry_threshold:
                direction = Direction.SHORT
                is_entry = prev_direction != Direction.SHORT
            else:
                direction = prev_direction  # hold existing
                is_entry = False

            signals.append(Signal(direction, z, ts, is_entry=is_entry))
            prev_direction = direction

        return pd.Series(signals, index=spread.index, name="signal")
