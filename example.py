"""
example.py — End-to-end usage example with synthetic crack spread data.

Demonstrates the full pipeline:
    1. Generate synthetic cointegrated price series (crude oil + heating oil)
    2. Run walk-forward backtest with all four hedge ratio estimators
    3. Compare risk metrics across estimators
    4. Grid search on ZScoreSignal thresholds
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from spreadpy.data import PriceTimeSeries, TransactionCosts
from spreadpy.spread import (
    ConstantOLS,
    RollingOLS,
    KalmanFilter,
    KalmanFilterWithVelocity,
)
from spreadpy.signal import ZScoreSignal, CopulaSignal
from spreadpy.sizing import PositionSizer
from spreadpy.engine import WalkForwardEngine


# ---------------------------------------------------------------------------
# 1. Synthetic data generation (cointegrated pair)
# ---------------------------------------------------------------------------

def generate_synthetic_crack_spread(
    n: int = 1000,
    seed: int = 42,
) -> tuple[PriceTimeSeries, PriceTimeSeries]:
    """
    Simulate crude oil (x) and heating oil (y = β·x + spread).
    The spread follows an AR(1) / Ornstein-Uhlenbeck process.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)

    # Crude oil: geometric Brownian motion
    log_ret_x = rng.normal(0.0001, 0.015, n)
    x_vals = 70.0 * np.exp(np.cumsum(log_ret_x))

    # True hedge ratio drifts slowly (realistic for energy spreads)
    beta_true = 0.85 + 0.10 * np.sin(np.linspace(0, 2 * np.pi, n))

    # Spread: mean-reverting OU process
    theta_ou = 0.08     # mean-reversion speed
    sigma_ou = 0.30     # spread volatility
    spread_vals = np.zeros(n)
    for t in range(1, n):
        spread_vals[t] = (
            spread_vals[t - 1] * (1 - theta_ou)
            + rng.normal(0, sigma_ou)
        )

    y_vals = beta_true * x_vals + spread_vals

    ts_y = PriceTimeSeries(pd.Series(y_vals, index=dates), name="heating_oil")
    ts_x = PriceTimeSeries(pd.Series(x_vals, index=dates), name="crude_oil")
    return ts_y, ts_x


# ---------------------------------------------------------------------------
# 2. Run walk-forward for all estimators
# ---------------------------------------------------------------------------

def run_all_estimators(
    y: PriceTimeSeries,
    x: PriceTimeSeries,
) -> dict:
    costs   = TransactionCosts(slippage_bps=0.0, commission_pct=0.0)
    sizer   = PositionSizer(max_notional=50_000)
    signal  = ZScoreSignal(window=40, entry_threshold=0.75, exit_threshold=0)

    estimators = {
        "ConstantOLS":             ConstantOLS(),
        "RollingOLS(60)":          RollingOLS(window=60),
        "KalmanFilter":            KalmanFilter(alpha=1e-5),
        "KalmanFilterWithVelocity": KalmanFilterWithVelocity(alpha=1e-6),
    }

    results = {}
    for name, estimator in estimators.items():
        print(f"  Running {name}...", end=" ", flush=True)
        engine = WalkForwardEngine(
            estimator=estimator,
            signal_gen=signal,
            sizer=sizer,
            costs=costs,
            initial_capital=500_000,
            train_size=int(len(x)*0.75),
            test_size=int(len(x)*0.25),
        )
        folds = engine.run(y, x)
        agg   = engine.aggregate_results(folds)
        results[name] = {"folds": folds, "metrics": agg}
        sr  = agg.get("sharpe",       float("nan"))
        mdd = agg.get("max_drawdown", float("nan"))
        print(f"Sharpe={sr:.2f}, MaxDD={mdd:.1%}")

    return results


# ---------------------------------------------------------------------------
# 3. Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(results: dict) -> pd.DataFrame:
    rows = {}
    for name, res in results.items():
        m = res["metrics"]
        rows[name] = {
            "Sharpe":           round(m.get("sharpe", np.nan), 3),
            "Sortino":          round(m.get("sortino", np.nan), 3),
            "Ann. Return":      f"{m.get('annualised_return', np.nan):.1%}",
            "Volatility":       f"{m.get('volatility', np.nan):.1%}",
            "Max Drawdown":     f"{m.get('max_drawdown', np.nan):.1%}",
            "Calmar":           round(m.get("calmar", np.nan), 3),
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 4. Grid search on ZScoreSignal
# ---------------------------------------------------------------------------

def run_grid_search(
    y: PriceTimeSeries,
    x: PriceTimeSeries,
) -> pd.DataFrame:
    costs  = TransactionCosts(slippage_bps=2.0, commission_pct=0.0001)
    sizer  = PositionSizer(max_notional=50_000)
    signal = ZScoreSignal()   # defaults overridden by grid

    engine = WalkForwardEngine(
        estimator=KalmanFilter(alpha=1e-5),
        signal_gen=signal,
        sizer=sizer,
        costs=costs,
        initial_capital=500_000,
        train_size=200,
        test_size=50,
    )

    grid = {
        "window":           [30, 60, 90],
        "entry_threshold":  [0.75, 1.0, 1.5, 2.0],
        "exit_threshold":   [0.0, 0.3],
    }

    print("  Running grid search...", flush=True)
    df = engine.optimize_params(y, x, param_grid=grid, metric="sharpe")
    return df


# ---------------------------------------------------------------------------
# 5. Copula signal example
# ---------------------------------------------------------------------------

def run_copula_example(
    y: PriceTimeSeries,
    x: PriceTimeSeries,
) -> None:
    costs  = TransactionCosts(slippage_bps=2.0, commission_pct=0.0001)
    sizer  = PositionSizer(max_notional=50_000)
    signal = CopulaSignal(family="gaussian", entry_prob=0.10)

    engine = WalkForwardEngine(
        estimator=KalmanFilterWithVelocity(alpha=1e-6),
        signal_gen=signal,
        sizer=sizer,
        costs=costs,
        initial_capital=500_000,
        train_size=200,
        test_size=50,
    )

    folds = engine.run(y, x)
    agg   = engine.aggregate_results(folds)
    sr    = agg.get("sharpe",       float("nan"))
    mdd   = agg.get("max_drawdown", float("nan"))
    print(f"  Copula (Gaussian) + KalmanWithVelocity: Sharpe={sr:.2f}, MaxDD={mdd:.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Spread Trading Backtest — Example")
    print("=" * 60)

    # print("\n[1] Generating synthetic crack spread data (n=1000 bars)...")
    # y, x = generate_synthetic_crack_spread(n=1000)
    import yfinance as yf
    cl_raw = yf.download("CL=F", period="730d", interval="1h").Close['CL=F']
    cl_raw.index = cl_raw.index.tz_convert("UTC")
    cl = PriceTimeSeries(cl_raw, name="crude_oil")

    bz_raw = yf.download("HO=F", period="730d", interval="1h").Close['HO=F']
    bz_raw.index = bz_raw.index.tz_convert("UTC")
    bz = PriceTimeSeries(bz_raw, name="brent_oil")

    y_al, x_al = cl.align(bz)
    n_total = len(y_al)
    train_size = int(n_total * 0.7)
    test_size  = n_total - train_size
    costs   = TransactionCosts(slippage_bps=0, commission_pct=0)
    sizer   = PositionSizer(max_notional=50_000)
    signal  = ZScoreSignal(window=60, entry_threshold=2, exit_threshold=0)
    engine = WalkForwardEngine(
        estimator=KalmanFilterWithVelocity(alpha=1e-6),
        signal_gen=signal,
        sizer=sizer,
        costs=costs,
        initial_capital=500_000,
        train_size=train_size,
        test_size=test_size,
        periods_per_year=252*23,  # hourly data (6.5 trading hours per day)
    )
    folds = engine.run(cl, bz)
    agg   = engine.aggregate_results(folds)
    sr  = agg.get("sharpe",       float("nan"))
    mdd = agg.get("max_drawdown", float("nan"))

    result = folds[0]
    eq = result.equity_curve["equity"]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                            gridspec_kw={"height_ratios": [3, 1, 1]})

    # ── Panel 1 : PnL cumulé ─────────────────────────────────────────────
    pnl = eq - eq.iloc[0]   # en dollars, base 0
    ax = axes[0]
    ax.plot(pnl.index, pnl, color="#378ADD", linewidth=1.2)
    ax.axhline(0, color="#888780", linewidth=0.5, linestyle="--")
    ax.fill_between(pnl.index, pnl, 0,
                    where=(pnl >= 0), alpha=0.1, color="#1D9E75")
    ax.fill_between(pnl.index, pnl, 0,
                    where=(pnl <  0), alpha=0.1, color="#E24B4A")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title(f"Sharpe={sr:.2f}  |  MaxDD={mdd:.1%}  |  "
                f"Train: {result.train_end.date()}  →  Test: {result.test_start.date()} / {result.test_end.date()}")

    # ── Panel 2 : drawdown ───────────────────────────────────────────────
    dd = (eq / eq.cummax() - 1) * 100
    axes[1].fill_between(dd.index, dd, 0, color="#E24B4A", alpha=0.4)
    axes[1].plot(dd.index, dd, color="#E24B4A", linewidth=0.8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].axhline(0, color="#888780", linewidth=0.5)

    # ── Panel 3 : z-score + signaux ──────────────────────────────────────
    signals = result.signals
    zs = pd.Series([s.zscore for s in signals], index=signals.index)
    dirs = pd.Series([int(s.direction) for s in signals], index=signals.index)

    axes[2].plot(zs.index, zs, color="#888780", linewidth=0.7, alpha=0.8)
    axes[2].axhline(0,     color="#888780", linewidth=0.5)
    axes[2].axhline( 0.75, color="#1D9E75", linewidth=0.8, linestyle="--")
    axes[2].axhline(-0.75, color="#1D9E75", linewidth=0.8, linestyle="--")

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
    




    # print("\n[2] Walk-forward backtest — all estimators (ZScore signal):")
    # results = run_all_estimators(cl, bz)

    # print("\n[3] Comparison table:")
    # table = build_comparison_table(results)
    # print(table.to_string())

    # print("\n[4] Grid search on ZScoreSignal (KalmanFilter estimator):")
    # grid_df = run_grid_search(cl, bz)
    # print(grid_df[["window", "entry_threshold", "exit_threshold", "sharpe", "max_drawdown"]].head(10).to_string(index=False))

    # print("\n[5] Copula signal example:")
    # run_copula_example(cl, bz)

    # print("\nDone.")
