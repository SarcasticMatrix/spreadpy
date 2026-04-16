from typing import Optional

import numpy as np
import pandas as pd

from spreadpy.signal.signal import SignalGenerator, Signal, Direction
from spreadpy.spread.spreadSeries import SpreadSeries


class ZScoreSignal(SignalGenerator):
    """
    Classic z-score entry/exit signal.

    Entry:
      - LONG  spread when zscore < -entry_threshold
      - SHORT spread when zscore > +entry_threshold

    Exit:
      - FLAT when |zscore| < exit_threshold  OR  |zscore| > stop_threshold

    Position sizing is driven downstream by the zscore value.

    Parameters
    ----------
    window          : Rolling window for z-score computation (bars)
    entry_threshold : |z| above which we enter
    exit_threshold  : |z| below which we exit
    stop_threshold  : |z| above which we stop-out (risk management)
    """

    def __init__(
        self,
        window: int = 60,
        entry_threshold: float = 1.0,
        exit_threshold: float = 0.0,
        stop_threshold: float = 4.0,
    ) -> None:
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_threshold = stop_threshold

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

            z_abs = abs(z)

            # --- Exit / stop conditions (checked first) ---
            if prev_direction != Direction.FLAT:
                if z_abs < self.exit_threshold or z_abs > self.stop_threshold:
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
