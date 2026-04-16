"""
signal.py — Layer 3: Signal generation
Signal (dataclass), SignalGenerator (abstract),
ZScoreSignal, CopulaSignal
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from spreadpy.spread.spreadSeries import SpreadSeries


class Direction(IntEnum):
    LONG  =  1   # long spread: buy y, sell x
    SHORT = -1   # short spread: sell y, buy x
    FLAT  =  0   # no position / exit


@dataclass
class Signal:
    """
    Output of a SignalGenerator at a single bar.

    Attributes
    ----------
    direction   : Direction (LONG / SHORT / FLAT)
    zscore      : Z-score of the spread at signal time
    prob        : Conditional probability of mean-reversion (copula) or NaN
    timestamp   : Bar timestamp
    is_entry    : True for entry signals, False for exit / hold
    """

    direction: Direction
    zscore: float
    timestamp: pd.Timestamp
    prob: float = float("nan")
    is_entry: bool = False

    def __repr__(self) -> str:
        p = f"{self.prob:.3f}" if not np.isnan(self.prob) else "n/a"
        return (
            f"Signal({self.timestamp.date()} "
            f"{self.direction.name:5s} "
            f"z={self.zscore:+.2f} p={p})"
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SignalGenerator(ABC):
    """
    Abstract signal generator.

    Subclasses implement `generate(spread)` which returns a
    pd.Series of Signal objects indexed by timestamp.

    fit() must be called on in-sample data before generate() is
    called on out-of-sample data (walk-forward discipline).
    """

    @abstractmethod
    def fit(self, spread: SpreadSeries) -> "SignalGenerator":
        """Calibrate parameters on in-sample spread."""

    @abstractmethod
    def generate(self, spread: SpreadSeries) -> pd.Series:
        """
        Generate signals on (possibly out-of-sample) spread.
        Returns pd.Series of Signal objects, indexed by spread.index.
        """

