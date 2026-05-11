"""
grid_search.py — Grid search over entry_threshold, revert_threshold and window.

Pipeline:
    1. Download price data
    2. Run BacktestEngine with log_prices=True for each (entry, revert, window)
    3. Print pivot tables (Sharpe, MaxDD, Sortino, Profit Factor, Hit Rate, # trades)
    4. Plot heatmaps — one row per window, one column per metric
    5. Save figure as PDF
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
WINDOWS = [15, 30, 60, 90]

# Metrics to pivot / plot  →  (key, display label, colormap, higher-is-better)
METRICS = [
    ("sharpe",        "Sharpe ratio",   "RdYlGn", True),
    ("max_drawdown",  "Max drawdown",   "RdYlGn", True),   # less negative = better
    ("sortino",       "Sortino ratio",  "RdYlGn", True),
    ("profit_factor", "Profit factor",  "RdYlGn", True),
    ("win_rate",      "Hit rate",       "RdYlGn", True),
    ("n_trades",      "# trades",       "Blues",  None),   # informational
]


if __name__ == "__main__":

    # ── 1. Data (downloaded once) ─────────────────────────────────────────────
    print("Downloading data…")
    long = PriceTimeSeries(fetch_history("ZW=F", period="730d", interval="1h"), name="long")
    short = PriceTimeSeries(fetch_history("ZM=F", period="730d", interval="1h"), name="short")

    # ── 2. Grid search ────────────────────────────────────────────────────────
    combos = [
        (entry, revert, window)
        for entry, revert, window in itertools.product(ENTRY_THRESHOLDS, REVERT_THRESHOLDS, WINDOWS)
        if revert < entry       # revert must be strictly below entry
    ]

    rows = []
    n = len(combos)
    for i, (entry, revert, window) in enumerate(combos, 1):
        print(f"  [{i:>3}/{n}]  entry={entry:.2f}  revert={revert:.2f}  window={window}", end="\r")

        engine = BacktestEngine(
            estimator=KalmanFilterWithVelocity(alpha=1e-6),
            signal_gen=ZScoreSignal(
                window=window,
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

        _, result = engine.run(long, short)
        rows.append({
            "entry_threshold":  entry,
            "revert_threshold": revert,
            "window":           window,
            **result.metrics.to_dict(),
        })

    print(f"\n  Done — {n} combinations evaluated.\n")
    results_df = pd.DataFrame(rows)

    # ── 3. Pivot tables ───────────────────────────────────────────────────────
    pd.set_option("display.float_format", "{:+.3f}".format)
    for window in WINDOWS:
        df_w = results_df[results_df["window"] == window]
        print(f"\n{'='*60}")
        print(f"  Window = {window}")
        print(f"{'='*60}")
        for key, label, _, _ in METRICS:
            if key not in df_w.columns:
                continue
            pivot = df_w.pivot(
                index="entry_threshold",
                columns="revert_threshold",
                values=key,
            )
            pivot.index.name   = "entry \\ revert"
            pivot.columns.name = None
            print(f"\n── {label} ──")
            print(pivot.to_string())
        print()

    # ── 4. Heatmaps ───────────────────────────────────────────────────────────
    available_metrics = [(k, lbl, cm, hib) for k, lbl, cm, hib in METRICS if k in results_df.columns]
    n_metrics = len(available_metrics)
    n_windows = len(WINDOWS)

    cell_w, cell_h = 6, 5.5
    fig, axes = plt.subplots(
        n_windows, n_metrics,
        figsize=(cell_w * n_metrics, cell_h * n_windows),
        sharey="row",
    )
    # Ensure axes is always 2-D
    if n_windows == 1:
        axes = axes[np.newaxis, :]
    if n_metrics == 1:
        axes = axes[:, np.newaxis]

    for row_idx, window in enumerate(WINDOWS):
        df_w = results_df[results_df["window"] == window]

        for col_idx, (key, label, cmap, higher_is_better) in enumerate(available_metrics):
            ax = axes[row_idx, col_idx]

            pivot = df_w.pivot(
                index="entry_threshold",
                columns="revert_threshold",
                values=key,
            )
            vals = pivot.values.astype(float)

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
            if col_idx == 0:
                ax.set_ylabel(f"window={window}\nentry_threshold", fontsize=8)
            else:
                ax.set_ylabel("entry_threshold", fontsize=8)
            ax.set_title(label, fontsize=9)

            fmt = "{:.0f}" if key == "n_trades" else "{:.2f}"
            for r in range(pivot.shape[0]):
                for c in range(pivot.shape[1]):
                    v = vals[r, c]
                    if not np.isnan(v):
                        ax.text(c, r, fmt.format(v),
                                ha="center", va="center", fontsize=7,
                                color="black")

    fig.suptitle(
        "Grid search — entry_threshold × revert_threshold × window  (log-prices)",
        fontsize=12,
        y=1.002,
    )
    plt.tight_layout()

    pdf_path = "grid_search.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Figure saved to {pdf_path}")

    plt.show()
