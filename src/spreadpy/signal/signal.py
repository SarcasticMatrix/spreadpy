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
    """
    Spread position direction.

    ``LONG``  (+1): buy y, sell x  — spread expected to rise (revert upward).
    ``SHORT`` (−1): sell y, buy x  — spread expected to fall (revert downward).
    ``FLAT``   (0): no position / exit signal.
    """
    LONG  =  1
    SHORT = -1
    FLAT  =  0


@dataclass
class Signal:
    """
    Output of a SignalGenerator at a single bar.

    :param Direction direction: Desired position direction (LONG, SHORT, or FLAT).
    :param float zscore: Z-score of the spread at signal time, used for position sizing.
    :param pd.Timestamp timestamp: Bar timestamp.
    :param float prob: Conditional mean-reversion probability from a copula signal,
        or ``nan`` for z-score signals.
    :param bool is_entry: True for new position entries; False for holds or exits.
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
    Abstract base class for signal generators.

    Enforces the fit / generate discipline required for walk-forward backtesting:
    :meth:`fit` is called on in-sample data to calibrate any parameters,
    then :meth:`generate` is called on out-of-sample data to produce signals
    without lookahead.
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

