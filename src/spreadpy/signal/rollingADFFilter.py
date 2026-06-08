from dataclasses import replace

import pandas as pd

from spreadpy.signal.signal import Direction, Signal, SignalGenerator
from spreadpy.spread.spreadSeries import SpreadSeries


class RollingADFFilter(SignalGenerator):
    """
    Wraps any :class:`SignalGenerator` and suppresses new entries when the
    spread fails a rolling ADF stationarity test.

    At each bar t, the ADF test is evaluated on the preceding ``adf_window``
    bars of the spread residuals. A new position is only opened if the
    p-value is below ``p_threshold`` (i.e. the spread is stationary).

    Exits and holds are always passed through unchanged. If an entry is
    blocked and the spread becomes stationary while still in the entry zone,
    the position opens at the first bar where the confirmation condition is
    met — with ``is_entry=True`` so sizers compute a fresh size.

    To reduce false signals from a noisy p-value, set ``min_confirm > 1``:
    entry is only allowed after ``min_confirm`` consecutive bars where the
    p-value stayed below ``p_threshold``.

    Usage::

        signal_gen = RollingADFFilter(
            base=ZScoreSignal(window=60, entry_threshold=1.5),
            adf_window=120,
            p_threshold=0.05,
            min_confirm=3,
        )

    :param SignalGenerator base: Underlying signal generator to wrap.
    :param int adf_window: Number of bars for the rolling ADF window.
    :param float p_threshold: Maximum p-value to allow an entry (default 0.05).
    :param int min_confirm: Number of consecutive bars the p-value must stay
        *above* ``p_threshold`` to block an entry (default 1). With the
        default of 1, any bar above the threshold blocks entry. Increase to
        require sustained non-stationarity before blocking — a single noisy
        bar above the threshold will not prevent entry.
    """

    def __init__(
        self,
        base: SignalGenerator,
        adf_window: int = 120,
        p_threshold: float = 0.05,
        min_confirm: int = 1,
    ) -> None:
        self.base = base
        self.adf_window = adf_window
        self.p_threshold = p_threshold
        self.min_confirm = min_confirm

    def fit(self, spread: SpreadSeries) -> "RollingADFFilter":
        self.base.fit(spread)
        return self

    def generate(self, spread: SpreadSeries) -> pd.Series:
        base_signals = self.base.generate(spread)
        adf_pvalues = spread.rolling_adf(self.adf_window)

        filtered: list[Signal] = []
        our_prev = Direction.FLAT
        consecutive_above = 0

        for ts, sig in base_signals.items():
            pval = adf_pvalues.at[ts]
            if pd.isna(pval):
                pval = 1.0

            if pval > self.p_threshold:
                consecutive_above += 1
            else:
                consecutive_above = 0

            blocked = consecutive_above >= self.min_confirm
            is_new_entry = our_prev == Direction.FLAT and sig.direction != Direction.FLAT

            if is_new_entry and blocked:
                # Non-stationarity confirmed — block entry, stay flat.
                filtered.append(Signal(Direction.FLAT, sig.zscore, ts, is_entry=False))
            elif is_new_entry:
                # Confirmed stationary: force is_entry=True for correct sizing.
                filtered.append(replace(sig, is_entry=True))
                our_prev = sig.direction
            else:
                filtered.append(sig)
                our_prev = sig.direction

        return pd.Series(filtered, index=spread.index, name="signal")
