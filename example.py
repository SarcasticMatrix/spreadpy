"""
example.py — End-to-end usage example: CL/HO crack spread backtest.

Pipeline:
    1. Download ~6 years of hourly CL=F / HO=F data (3 × 730-day chunks)
    2. Run a single walk-forward fold with KalmanFilterWithVelocity + ZScoreSignal
    3. Plot cumulative PnL, drawdown, and z-score signals
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils import fetch_history
from spreadpy.data import PriceTimeSeries
from spreadpy.spread import KalmanFilterWithVelocity, KalmanFilter
from spreadpy.signal import ZScoreSignal
from spreadpy.sizing import KellySizer, LinearSizer
from spreadpy.backtest import TransactionCosts, BacktestEngine


if __name__ == "__main__":
    print("=" * 60)
    print("Spread Trading Backtest — CL/HO Crack Spread")
    print("=" * 60)

    # ── 1. Data ──────────────────────────────────────────────────────────
    cl = PriceTimeSeries(fetch_history("CC=F", period="730d", interval="1h"), name="crude_oil")
    ho = PriceTimeSeries(fetch_history("KC=F", period="730d", interval="1h"), name="heating_oil")

    # ── 2. Backtest ──────────────────────────────────────────────────────
    entry_threshold  = 1
    revert_threshold = 0.5
    f_max = 0.25
    engine = BacktestEngine(
        estimator=KalmanFilterWithVelocity(alpha=1e-6),
        signal_gen=ZScoreSignal(window=60, entry_threshold=entry_threshold, revert_threshold=revert_threshold),
        sizer=KellySizer(z0=entry_threshold, z_revert=revert_threshold, f_max=f_max),
        costs=TransactionCosts(slippage_bps=1.0, commission_bps=1.0),
        initial_capital=500_000,
        train_frac=0.7,
        val_frac=0.0,           # pas de validation, train/test seulement
        periods_per_year=252 * 10,
        log_prices=True,        # Kalman on log-prices (homoscedastic σ²_ε)
    )

    _, result = engine.run(cl, ho)   # val_result is None (val_frac=0)
    sr  = result.metrics.get("sharpe",       float("nan"))
    mdd = result.metrics.get("max_drawdown", float("nan"))
    eq  = result.equity_curve["equity"]

    result.print_summary()

    # ── 3. Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    # Integer x-axis to remove weekend / overnight gaps.
    # All series must be mapped to positions in eq.index.
    xi_map = {ts: i for i, ts in enumerate(eq.index)}
    xi_eq  = np.arange(len(eq))

    def xi(index):
        """Map a DatetimeIndex to integer positions via xi_map."""
        return np.array([xi_map[ts] for ts in index if ts in xi_map])

    def xv(series):
        """Values of `series` restricted to timestamps present in xi_map."""
        return series.loc[series.index.isin(xi_map)].values

    # Panel 1 — cumulative PnL + mark-to-market
    pnl   = eq - eq.iloc[0]
    mtm   = result.equity_curve["unrealised_pnl"]
    ax    = axes[0]
    ax.plot(xi_eq, pnl, color="#378ADD", linewidth=1.2, label="Total PnL")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.fill_between(xi_eq, pnl, 0, where=(pnl >= 0), alpha=0.1, color="#1D9E75")
    ax.fill_between(xi_eq, pnl, 0, where=(pnl <  0), alpha=0.1, color="#E24B4A")
    ax.set_ylabel("PnL ($)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(
        f"Sharpe={sr:.2f}  |  MaxDD={mdd:.1%}  |  "
        f"Train: {result.train_end.date()}  →  "
        f"Test: {result.eval_start.date()} / {result.eval_end.date()}"
    )

    # Panel 2 — drawdown
    dd = (eq / eq.cummax() - 1) * 100
    axes[1].fill_between(xi_eq, dd, 0, color="#E24B4A", alpha=0.4)
    axes[1].plot(xi_eq, dd, color="#E24B4A", linewidth=0.8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].axhline(0, color="#888780", linewidth=0.5)

    # Panel 3 — z-score + entry signals
    signals  = result.signals
    zs       = pd.Series([s.zscore         for s in signals], index=signals.index)
    dirs     = pd.Series([int(s.direction) for s in signals], index=signals.index)
    entries  = pd.Series([s.is_entry       for s in signals], index=signals.index)

    long_idx  = entries & (dirs ==  1)
    short_idx = entries & (dirs == -1)
    prev_dirs = dirs.shift(1, fill_value=0)
    flat_idx  = (dirs == 0) & (prev_dirs != 0)

    axes[2].plot(xi(zs.index), xv(zs), color="gray", linewidth=1.2, alpha=0.6)
    axes[2].axhline(revert_threshold, color="blue", linewidth=0.5)
    axes[2].axhline( entry_threshold, color="blue", linewidth=0.8, linestyle="--")
    axes[2].axhline(-entry_threshold, color="blue", linewidth=0.8, linestyle="--")
    axes[2].scatter(xi(zs.index[long_idx]),  xv(zs[long_idx]),
                    marker="^", color="#1D9E75", s=80, zorder=5, label="Long entry")
    axes[2].scatter(xi(zs.index[short_idx]), xv(zs[short_idx]),
                    marker="v", color="#E24B4A", s=80, zorder=5, label="Short entry")
    axes[2].scatter(xi(zs.index[flat_idx]),  xv(zs[flat_idx]),
                    marker="s", color="#888780", s=50, zorder=5, label="Exit")
    axes[2].set_ylabel("Z-score")
    axes[2].legend(fontsize=8, ncol=3)

    # Panel 4 — spread quantity over time
    y_changes = pd.Series(
        [t.direction * t.qty for t in result.trades if t.leg == "y"],
        index=pd.DatetimeIndex([t.timestamp for t in result.trades if t.leg == "y"]),
    ).groupby(level=0).sum()
    spread_qty = y_changes.reindex(eq.index, fill_value=0).cumsum().ffill()

    axes[3].step(xi_eq, spread_qty, where="post", color="#378ADD", linewidth=1.0)
    axes[3].fill_between(xi_eq, spread_qty, 0,
                         where=(spread_qty > 0), step="post",
                         color="#1D9E75", alpha=0.3, label="Long spread")
    axes[3].fill_between(xi_eq, spread_qty, 0,
                         where=(spread_qty < 0), step="post",
                         color="#E24B4A", alpha=0.3, label="Short spread")
    axes[3].axhline(0, color="#888780", linewidth=0.5)
    axes[3].set_ylabel("Spread qty (units of y)")
    axes[3].legend(fontsize=8, ncol=2)

    # Date formatter on the shared x-axis (bottom panel only)
    def _date_fmt(x, *_):
        i = int(round(x))
        if 0 <= i < len(eq.index):
            return eq.index[i].strftime("%b %d\n%Y")
        return ""
    axes[-1].xaxis.set_major_formatter(mticker.FuncFormatter(_date_fmt))
    axes[-1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()



    # ── 4. Spread comparison: β_t (dynamic) vs β_entry (frozen) ─────────
    # result.spread is in log-space (log_prices=True):
    #   s_t = log(y_t) − β_t · log(x_t)
    spread_bt = result.spread.residuals
    beta_t    = result.spread.hedge_ratio_ts
    y_ser     = result.spread.y.series   # log(y)
    x_ser     = result.spread.x.series   # log(x)

    # Reconstruct β_entry: β frozen when entering a position, NaN while flat
    dirs_full = dirs.reindex(spread_bt.index, fill_value=0)
    active_beta = np.nan
    prev_dir    = 0
    beta_entry_vals = []
    for ts, d in dirs_full.items():
        d = int(d)
        if d != 0 and (prev_dir == 0 or d != prev_dir):   # entry or flip
            active_beta = float(beta_t.loc[ts])
        elif d == 0:                                        # back to flat
            active_beta = np.nan
        beta_entry_vals.append(active_beta)
        prev_dir = d

    # ffill so the last known β_entry is always visible (even when flat)
    beta_entry_ts = pd.Series(beta_entry_vals, index=spread_bt.index).ffill().fillna(beta_t)
    spread_entry  = y_ser - beta_entry_ts * x_ser

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Panel 1 — both spreads overlaid
    ax2 = axes2[0]
    ax2.plot(xi(spread_bt.index),    xv(spread_bt),    color="#378ADD", linewidth=1.0,
             alpha=0.9, label="β_t  (dynamic)")
    ax2.plot(xi(spread_entry.index), xv(spread_entry), color="#9B59B6", linewidth=1.0,
             alpha=0.7, label="β_entry (frozen)")
    ax2.grid(linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5)


    sp_at_bt = spread_bt.reindex(zs.index)
    ax2.scatter(xi(zs.index[long_idx]),  xv(sp_at_bt[long_idx]),
                marker="^", color="#1D9E75", s=80, zorder=5, label="Long entry")
    ax2.scatter(xi(zs.index[short_idx]), xv(sp_at_bt[short_idx]),
                marker="v", color="#E24B4A", s=80, zorder=5, label="Short entry")
    ax2.scatter(xi(zs.index[flat_idx]),  xv(sp_at_bt[flat_idx]),
                marker="s", color="#888780", s=50, zorder=5, label="Exit")
    ax2.set_ylabel("log(y) − β·log(x)")
    ax2.set_title("Log-spread — β_t (dynamic) vs β_entry (frozen at entry)", fontsize=9)
    ax2.legend(fontsize=8, ncol=5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=8)

    # Panel 2 — β_t vs β_entry (divergence)
    ax2 = axes2[1]
    ax2.plot(xi(beta_t.index),         xv(beta_t),         color="#378ADD", linewidth=1.0,
             alpha=0.9, label="β_t  (dynamic)")
    ax2.plot(xi(beta_entry_ts.index),  xv(beta_entry_ts),  color="#9B59B6", linewidth=1.0,
             alpha=0.7, label="β_entry (frozen)")
    ax2.set_ylabel("β")
    ax2.set_title("Hedge ratio — β_t vs β_entry", fontsize=9)
    ax2.legend(fontsize=8, ncol=2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=8)
    ax2.grid(linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5)

    axes2[-1].xaxis.set_major_formatter(mticker.FuncFormatter(_date_fmt))
    axes2[-1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))

    fig2.suptitle("β drift effect — spread & hedge ratio comparison", fontsize=10)
    plt.tight_layout()
    plt.show()
