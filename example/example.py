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
from spreadpy.spread import KalmanFilterWithVelocity, KalmanFilter, ConstantOLS
from spreadpy.signal import ZScoreSignal
from spreadpy.sizing import InverseVolSizer, KellyTruncatedEntry, KellyTruncatedExit, KellyTruncatedBoth, LinearSizer
from spreadpy.backtest import TransactionCosts, BacktestEngine


if __name__ == "__main__":
    print("=" * 60)
    print("Spread Trading Backtest — Spread")
    print("=" * 60)

    # ── 1. Data ──────────────────────────────────────────────────────────
    long = PriceTimeSeries(fetch_history("ZW=F", period="730d", interval="1h"), name="long")
    short = PriceTimeSeries(fetch_history("ZM=F", period="730d", interval="1h"), name="short")
    # long = PriceTimeSeries(fetch_history("KO", period="730d", interval="1h"), name="long")
    # short = PriceTimeSeries(fetch_history("PEP", period="730d", interval="1h"), name="short")
    # long = PriceTimeSeries(fetch_history("BZ=F", period="730d", interval="1h"), name="brent")
    # short = PriceTimeSeries(fetch_history("CL=F", period="730d", interval="1h"), name="wti")

    periods_per_year = long._series.groupby(long._series.index.year).count().mean()

    # ── 2. Backtest ──────────────────────────────────────────────────────
    entry_threshold  = 0.75
    revert_threshold = 0.1
    f_max = 0.25
    def scale_fn(abs_z: float) -> float:
        return 1-float(np.clip((abs_z - entry_threshold) / 3.0, 0.0, 1.0))
    engine = BacktestEngine(
        estimator=KalmanFilterWithVelocity(alpha=1e-4, alpha_dgam=1e-6, add_intercept=False),
        signal_gen=ZScoreSignal(window=60, entry_threshold=entry_threshold, revert_threshold=revert_threshold),
        sizer=KellyTruncatedExit(z_revert=revert_threshold, f_max=f_max),
        # sizer=LinearSizer(scale_fn=scale_fn),
        # sizer=InverseVolSizer(window=60, target_vol=0.1, f_max=0.5),
        adf_p_threshold=0.05,
        adf_window=24*2,
        costs=TransactionCosts(slippage_bps=0.0, commission_bps=0.0, min_commission=0),
        initial_capital=500_000,
        train_frac=0.4,
        val_frac=0.0,           # pas de validation, train/test seulement
        periods_per_year=periods_per_year,
        log_prices=True,        # Kalman on log-prices (homoscedastic σ²_ε)
    )

    _, result = engine.run(long, short)   # val_result is None (val_frac=0)
    sr  = result.metrics.get("sharpe",       float("nan"))
    mdd = result.metrics.get("max_drawdown", float("nan"))
    eq  = result.equity_curve["equity"]

    # Re-fit Kalman on full log-prices to expose dgamma for plotting.
    # The engine deep-copies its estimator internally so engine.estimator stays unfitted.
    _long_al, _short_al = long.align(short)
    _kf = KalmanFilterWithVelocity(alpha=1e-10, add_intercept=False)
    _kf.fit(
        PriceTimeSeries(np.log(_long_al.series), name=_long_al.name),
        PriceTimeSeries(np.log(_short_al.series), name=_short_al.name),
    )
    dgamma_ts = _kf.velocity_ts_.loc[result.eval_start : result.eval_end]

    result.print_summary()

    _adf_window    = getattr(engine.signal_gen, "adf_window",    120)
    _adf_threshold = getattr(engine.signal_gen, "p_threshold", 0.05)
    adf_pvalues    = result.spread.rolling_adf(_adf_window)

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

    # Panel 1 — cumulative PnL + drawdown overlay (dd=0 aligned with pnl.max())
    pnl = eq - eq.iloc[0]
    dd  = (eq / eq.cummax() - 1) * 100
    ax  = axes[0]
    ax.plot(xi_eq, pnl, color="black", linewidth=1.2, label="Total PnL")
    ax.fill_between(xi_eq, pnl, 0, where=(pnl >= 0), alpha=0.1, color="green")
    ax.fill_between(xi_eq, pnl, 0, where=(pnl <  0), alpha=0.1, color="red")
    ax.set_ylim(min(pnl.min(), 0), pnl.max())
    ax.set_ylabel("PnL ($)")

    ax_dd = ax.twinx()
    ax_dd.fill_between(xi_eq, dd, 0, color="red", alpha=0.2)
    ax_dd.plot(xi_eq, dd, color="red", linewidth=1, alpha=0.6, label="Drawdown")
    ax_dd.set_ylim(dd.min() * 1.5, 0)
    ax_dd.set_ylabel("Drawdown (%)", color="red", fontsize=8)
    ax_dd.tick_params(axis='y', labelcolor="red", labelsize=7)
    ax_dd.spines[["top"]].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_dd.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2)
    ax.set_title(
        f"Sharpe $=$ {sr:.2f}  |  MaxDD $=$ {mdd:.1%}  |  "
        f"Train: {result.train_end.date()}  →  "
        f"Test: {result.eval_start.date()} / {result.eval_end.date()}",
        loc='left', fontweight='bold',
    )

    # Panel 2 — rolling ADF p-value
    _adf_xi = xi(adf_pvalues.index)
    _adf_xv = xv(adf_pvalues)
    axes[1].plot(_adf_xi, _adf_xv, color="black", linewidth=1.0, alpha=0.9)
    axes[1].axhline(_adf_threshold, color="gray", linewidth=0.6, label=f"$p = {_adf_threshold}$")
    axes[1].fill_between(_adf_xi, _adf_xv, _adf_threshold,
                         where=(_adf_xv > _adf_threshold), alpha=0.15, color="red")
    axes[1].set_title(f"Rolling ADF p-value  (window = {_adf_window})",
                      loc='left', fontweight='bold', fontsize=9)
    axes[1].set_ylabel(r"ADF $p$-value")
    axes[1].legend(fontsize=8)

    # Panel 3 — z-score + entry signals
    signals  = result.signals
    zs       = pd.Series([s.zscore         for s in signals], index=signals.index)
    dirs     = pd.Series([int(s.direction) for s in signals], index=signals.index)
    entries  = pd.Series([s.is_entry       for s in signals], index=signals.index)

    long_idx  = entries & (dirs ==  1)
    short_idx = entries & (dirs == -1)
    prev_dirs = dirs.shift(1, fill_value=0)
    flat_idx  = (dirs == 0) & (prev_dirs != 0)

    axes[2].plot(xi(zs.index), xv(zs), color="black", linewidth=1)
    axes[2].axhline(revert_threshold, color="gray", linewidth=0.6)
    axes[2].axhline(-revert_threshold, color="gray", linewidth=0.6)
    axes[2].axhline( entry_threshold, color="gray", linewidth=0.6, linestyle="--")
    axes[2].axhline(-entry_threshold, color="gray", linewidth=0.6, linestyle="--")
    axes[2].scatter(xi(zs.index[long_idx]),  xv(zs[long_idx]),
                    marker="^", color="green", s=80, zorder=5, label="Long entry")
    axes[2].scatter(xi(zs.index[short_idx]), xv(zs[short_idx]),
                    marker="v", color="red", s=80, zorder=5, label="Short entry")
    axes[2].scatter(xi(zs.index[flat_idx]),  xv(zs[flat_idx]),
                    marker="s", color="gray", s=50, zorder=5, label="Exit")
    axes[2].set_title(r"z-score", loc='left', fontweight='bold', fontsize=9)
    axes[2].set_ylabel(r"$z_t$")
    axes[2].legend(fontsize=8, ncol=3)

    # Panel 4 — spread quantity over time
    y_changes = pd.Series(
        [t.direction * t.qty for t in result.trades if t.leg == "y"],
        index=pd.DatetimeIndex([t.timestamp for t in result.trades if t.leg == "y"]),
    ).groupby(level=0).sum()
    spread_qty = y_changes.reindex(eq.index, fill_value=0).cumsum().ffill()

    axes[3].step(xi_eq, spread_qty, where="post", color="black", linewidth=1.0)
    axes[3].fill_between(xi_eq, spread_qty, 0,
                         where=(spread_qty >= 0), step="post",
                         color="green", alpha=0.3, label="Long spread")
    axes[3].fill_between(xi_eq, spread_qty, 0,
                         where=(spread_qty <= 0), step="post",
                         color="red", alpha=0.3, label="Short spread")
    axes[3].axhline(0, color="gray", linewidth=0.6)
    axes[3].set_title(r"Spread quantity (units of $y$)", loc='left', fontweight='bold', fontsize=9)
    axes[3].set_ylabel(r"qty$_y$")
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

    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    # Panel 1 — both spreads overlaid
    ax2 = axes2[0]
    ax2.plot(xi(spread_bt.index),    xv(spread_bt),    color="blue", linewidth=1.0,
             alpha=0.9, label=r"$\beta_t$ (dynamic)")
    ax2.plot(xi(spread_entry.index), xv(spread_entry), color="orange", linewidth=1.0,
             alpha=0.7, label=r"$\beta_{\mathrm{entry}}$ (frozen)")
    ax2.grid(linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5)


    sp_at_bt = spread_bt.reindex(zs.index)
    ax2.scatter(xi(zs.index[long_idx]),  xv(sp_at_bt[long_idx]),
                marker="^", color="green", s=80, zorder=5, label="Long entry")
    ax2.scatter(xi(zs.index[short_idx]), xv(sp_at_bt[short_idx]),
                marker="v", color="red", s=80, zorder=5, label="Short entry")
    ax2.scatter(xi(zs.index[flat_idx]),  xv(sp_at_bt[flat_idx]),
                marker="s", color="gray", s=50, zorder=5, label="Exit")
    ax2.set_ylabel(r"$\log y_t - \beta_t \cdot \log x_t$")
    ax2.set_title(r"Log-spread — $\beta_t$ (dynamic) vs $\beta_{\mathrm{entry}}$ (frozen)", loc='left', fontweight='bold', fontsize=9)
    ax2.legend(fontsize=8, ncol=5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=8)

    # Panel 2 — β_t vs β_entry (divergence)
    ax2 = axes2[1]
    ax2.plot(xi(beta_t.index),         xv(beta_t),         color="blue", linewidth=1.0,
             alpha=0.9, label=r"$\beta_t$ (dynamic)")
    ax2.plot(xi(beta_entry_ts.index),  xv(beta_entry_ts),  color="orange", linewidth=1.0,
             alpha=0.7, label=r"$\beta_{\mathrm{entry}}$ (frozen)")
    ax2.set_ylabel(r"$\beta$")
    ax2.set_title(r"Hedge ratio — $\beta_t$ vs $\beta_{\mathrm{entry}}$", loc='left', fontweight='bold', fontsize=9)
    ax2.legend(fontsize=8, ncol=2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=8)
    ax2.grid(linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5)

    axes2[-1].xaxis.set_major_formatter(mticker.FuncFormatter(_date_fmt))
    axes2[-1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))

    # Panel 3 — γ̇_t (velocity / dgamma)
    ax2 = axes2[2]
    ax2.plot(xi(dgamma_ts.index), xv(dgamma_ts), color="blue", linewidth=1.0, alpha=0.9)
    ax2.axhline(0, color="#888780", linewidth=0.5)
    ax2.set_ylabel(r"$\dot{\gamma}_t$")
    ax2.set_title(r"Hedge ratio velocity $\dot{\gamma}_t$", loc='left', fontweight='bold', fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=8)
    ax2.grid(linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5)

    # Panel 4 — rolling ADF p-value
    ax2 = axes2[3]
    ax2.plot(xi(adf_pvalues.index), xv(adf_pvalues), color="blue", linewidth=1.0, alpha=0.9)
    ax2.axhline(_adf_threshold, color="black", linewidth=0.8, linestyle="--",
                label=f"$p = {_adf_threshold}$")
    ax2.set_ylabel(r"ADF $p$-value")
    ax2.set_title(f"Rolling ADF $p$-value  (window $= {_adf_window}$)", loc='left', fontweight='bold', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=8)
    ax2.grid(linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5)

    fig2.suptitle(r"$\beta$ drift effect — spread, hedge ratio & velocity", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()
