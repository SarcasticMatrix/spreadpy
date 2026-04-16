"""
example.py — End-to-end usage example: CL/HO crack spread backtest.

Pipeline:
    1. Download hourly CL=F / HO=F data via yfinance
    2. Run a single walk-forward fold with KalmanFilterWithVelocity + ZScoreSignal
    3. Plot cumulative PnL, drawdown, and z-score signals
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from spreadpy.data import PriceTimeSeries
from spreadpy.spread import KalmanFilterWithVelocity
from spreadpy.signal import ZScoreSignal
from spreadpy.sizing import PositionSizer
from spreadpy.backtest import TransactionCosts, BacktestEngine


if __name__ == "__main__":
    print("=" * 60)
    print("Spread Trading Backtest — CL/HO Crack Spread")
    print("=" * 60)

    # ── 1. Data ──────────────────────────────────────────────────────────
    cl_raw = yf.download("CL=F", period="730d", interval="1h").Close["CL=F"]
    cl_raw.index = cl_raw.index.tz_convert("UTC")
    cl = PriceTimeSeries(cl_raw, name="crude_oil")

    ho_raw = yf.download("HO=F", period="730d", interval="1h").Close["HO=F"]
    ho_raw.index = ho_raw.index.tz_convert("UTC")
    ho = PriceTimeSeries(ho_raw, name="heating_oil")

    # ── 2. Backtest ──────────────────────────────────────────────────────
    engine = BacktestEngine(
        estimator=KalmanFilterWithVelocity(alpha=1e-6),
        signal_gen=ZScoreSignal(window=60, entry_threshold=1, exit_threshold=0),
        sizer=PositionSizer(max_notional=50_000),
        costs=TransactionCosts(slippage_bps=0, commission_bps=0),
        initial_capital=500_000,
        train_frac=0.7,
        val_frac=0.0,           # pas de validation, train/test seulement
        periods_per_year=252 * 23,  # hourly (≈23 trading hours/day for futures)
    )

    _, result = engine.run(cl, ho)   # val_result is None (val_frac=0)
    sr  = result.metrics.get("sharpe",       float("nan"))
    mdd = result.metrics.get("max_drawdown", float("nan"))
    eq  = result.equity_curve["equity"]

    # ── 3. Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    # Panel 1 — cumulative PnL
    pnl = eq - eq.iloc[0]
    ax  = axes[0]
    ax.plot(pnl.index, pnl, color="#378ADD", linewidth=1.2)
    ax.axhline(0, color="#888780", linewidth=0.5, linestyle="--")
    ax.fill_between(pnl.index, pnl, 0, where=(pnl >= 0), alpha=0.1, color="#1D9E75")
    ax.fill_between(pnl.index, pnl, 0, where=(pnl <  0), alpha=0.1, color="#E24B4A")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title(
        f"Sharpe={sr:.2f}  |  MaxDD={mdd:.1%}  |  "
        f"Train: {result.train_end.date()}  →  "
        f"Test: {result.eval_start.date()} / {result.eval_end.date()}"
    )

    # Panel 2 — drawdown
    dd = (eq / eq.cummax() - 1) * 100
    axes[1].fill_between(dd.index, dd, 0, color="#E24B4A", alpha=0.4)
    axes[1].plot(dd.index, dd, color="#E24B4A", linewidth=0.8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].axhline(0, color="#888780", linewidth=0.5)

    # Panel 3 — z-score + signals
    signals = result.signals
    zs   = pd.Series([s.zscore    for s in signals], index=signals.index)
    dirs = pd.Series([int(s.direction) for s in signals], index=signals.index)

    axes[2].plot(zs.index, zs, color="#888780", linewidth=0.7, alpha=0.8)
    axes[2].axhline( 0,    color="#888780", linewidth=0.5)
    axes[2].axhline( 2.0,  color="#1D9E75", linewidth=0.8, linestyle="--")
    axes[2].axhline(-2.0,  color="#1D9E75", linewidth=0.8, linestyle="--")
    axes[2].scatter(signals.index[dirs ==  1], zs[dirs ==  1],
                    color="#1D9E75", s=10, zorder=5, label="Long")
    axes[2].scatter(signals.index[dirs == -1], zs[dirs == -1],
                    color="#D85A30", s=10, zorder=5, label="Short")
    axes[2].scatter(signals.index[dirs ==  0], zs[dirs ==  0],
                    color="#888780", s=4,  zorder=4, alpha=0.3, label="Flat")
    axes[2].set_ylabel("Z-score")
    axes[2].legend(fontsize=8, ncol=3)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()
