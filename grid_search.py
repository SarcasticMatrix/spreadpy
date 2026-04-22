"""
grid_search.py — Grid search over entry_threshold and revert_threshold.

Pipeline:
    1. Download CL=F / HO=F price data
    2. Run BacktestEngine with log_prices=True (Kalman on log-prices,
       trading on actual prices) for each (entry, revert) combination
    3. Print pivot tables (Sharpe, MaxDD, CDaR 5%, # trades)
    4. Plot heatmaps
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils import fetch_history
from spreadpy.data import PriceTimeSeries
from spreadpy.spread import KalmanFilterWithVelocity
from spreadpy.signal import ZScoreSignal, CopulaSignal
from spreadpy.sizing import KellyTruncatedEntry, LinearSizer, KellyTruncatedExit
from spreadpy.backtest import TransactionCosts, BacktestEngine


# ── Grid definition ───────────────────────────────────────────────────────────
ENTRY_THRESHOLDS  = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
REVERT_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.5]

# Metrics to pivot / plot  →  (display label, colormap, higher-is-better)
METRICS = [
    ("sharpe",       "Sharpe ratio",      "RdYlGn",   True),
    ("max_drawdown", "Max drawdown",      "RdYlGn",   True),   # less negative = better
    ("cdar_5",       "CDaR 5%",           "RdYlGn",   True),   # idem
    ("n_trades",     "# trades",          "Blues",    None),   # informational
]


if __name__ == "__main__":

    # ── 1. Data (downloaded once) ─────────────────────────────────────────────
    print("Downloading data…")
    cl = PriceTimeSeries(fetch_history("CC=F", period="730d", interval="1h"), name="crude_oil")
    ho = PriceTimeSeries(fetch_history("KC=F", period="730d", interval="1h"), name="heating_oil")

    # ── 2. Grid search ────────────────────────────────────────────────────────
    combos = [
        (entry, revert)
        for entry, revert in itertools.product(ENTRY_THRESHOLDS, REVERT_THRESHOLDS)
        if revert < entry       # revert must be strictly below entry
    ]

    rows = []
    n = len(combos)
    for i, (entry, revert) in enumerate(combos, 1):
        print(f"  [{i:>3}/{n}]  entry={entry:.2f}  revert={revert:.2f}", end="\r")

        engine = BacktestEngine(
            estimator=KalmanFilterWithVelocity(alpha=1e-6),
            signal_gen=ZScoreSignal(
                window=60,
                entry_threshold=entry,
                revert_threshold=revert,
            ),
            sizer=KellyTruncatedExit(z_revert=revert, f_max=0.25),
            costs=TransactionCosts(slippage_bps=2, commission_bps=3),
            initial_capital=500_000,
            train_frac=0.01,
            val_frac=0.0,
            periods_per_year=252 * 8,
            log_prices=True,
        )

        _, result = engine.run(cl, ho)
        rows.append({
            "entry_threshold":  entry,
            "revert_threshold": revert,
            **result.metrics.to_dict(),
        })

    print(f"\n  Done — {n} combinations evaluated.\n")
    results_df = pd.DataFrame(rows)

    # ── 3. Pivot tables ───────────────────────────────────────────────────────
    pd.set_option("display.float_format", "{:+.3f}".format)
    for key, label, _, _ in METRICS:
        if key not in results_df.columns:
            continue
        pivot = results_df.pivot(
            index="entry_threshold",
            columns="revert_threshold",
            values=key,
        )
        pivot.index.name   = "entry \\ revert"
        pivot.columns.name = None
        print(f"── {label} ──────────────────────────────")
        print(pivot.to_string())
        print()

    # ── 4. Heatmaps ───────────────────────────────────────────────────────────
    n_metrics = sum(1 for k, *_ in METRICS if k in results_df.columns)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    ax_iter = iter(axes)
    for key, label, cmap, higher_is_better in METRICS:
        if key not in results_df.columns:
            continue
        ax = next(ax_iter)

        pivot = results_df.pivot(
            index="entry_threshold",
            columns="revert_threshold",
            values=key,
        )
        vals = pivot.values.astype(float)

        # Symmetric normalisation around the midpoint for diverging colormaps
        if higher_is_better is not None:
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = None

        im = ax.imshow(vals, aspect="auto", cmap=cmap, norm=norm)
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=8)
        ax.set_xlabel("revert_threshold", fontsize=8)
        ax.set_ylabel("entry_threshold",  fontsize=8)
        ax.set_title(label, fontsize=9)

        # Annotate each cell
        fmt = "{:.0f}" if key == "n_trades" else "{:.2f}"
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                v = vals[r, c]
                if not np.isnan(v):
                    ax.text(c, r, fmt.format(v),
                            ha="center", va="center", fontsize=7,
                            color="black")

    fig.suptitle("Grid search — entry_threshold × revert_threshold  (log-prices)", fontsize=10)
    plt.tight_layout()
    plt.show()
