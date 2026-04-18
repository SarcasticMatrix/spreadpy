from __future__ import annotations


from itertools import combinations
from tqdm import tqdm

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller, coint

from spreadpy.data import PriceTimeSeries




class PairFinder:
    """
    Scan a universe of assets and rank candidate cointegrated pairs.

    Pipeline per pair:

    1. **NPD** — normalized price distance, cheap pre-filter.
    2. **Engle-Granger** cointegration test on aligned prices.
    3. **ADF** on OLS residuals to confirm spread stationarity.
    4. **Half-life** of mean reversion from AR(1) on the spread.

    Parameters
    ----------
    series : list[PriceTimeSeries]
        Universe of price series to scan.
    significance : float
        p-value threshold applied to both EG and ADF (default 0.05).
    npd_threshold : float
        Maximum NPD to pass pre-filter. Set to ``None`` to disable.
    """

    def __init__(
        self,
        series: list[PriceTimeSeries],
        significance: float = 0.05,
        npd_threshold: float | None = 0.3,
        log_prices: bool = False
    ) -> None:
        self.series = series
        self.significance = significance
        self.npd_threshold = npd_threshold
        self.log_prices = log_prices

        if self.log_prices:
            for i in range(len(self.series)):
                pts = self.series[i]
                self.series[i] = PriceTimeSeries(np.log(pts.series), name=pts.name)

    def scan(self) -> pd.DataFrame:
        """
        Scan all pairs and return a ranked DataFrame.

        Returns
        -------
        pd.DataFrame
            Sorted by EG p-value ascending, columns:
            ``x``, ``y``, ``npd``, ``eg_pvalue``, ``adf_pvalue``,
            ``half_life``, ``hedge_ratio``.
            Empty if no pair passes both significance thresholds.
        """
        rows = []

        for s1, s2 in tqdm(combinations(self.series, 2)):
            a, b = s1.align(s2)
            if len(a) < 30:
                continue

            npd = _npd(a.series, b.series)
            if self.npd_threshold is not None and npd > self.npd_threshold:
                continue

            eg_stat, eg_pvalue, _ = coint(a.series, b.series)
            hedge_ratio, residuals = _ols_residuals(a.series, b.series)
            adf_stat, adf_pvalue, _, _, adf_crit, _ = adfuller(residuals, autolag="AIC")

            if eg_pvalue >= self.significance or adf_pvalue >= self.significance:
                continue

            rows.append({
                "x":            a.name,
                "y":            b.name,
                "npd":          round(npd, 4),
                "eg_stat":      round(eg_stat, 4),
                "eg_pvalue":    round(eg_pvalue, 4),
                "adf_stat":     round(adf_stat, 4),
                "adf_pvalue":   round(adf_pvalue, 4),
                "half_life":    round(_half_life(residuals), 1),
                "hurst":        round(_hurst(residuals), 3),
                "adf_crit_1%":  round(adf_crit["1%"], 4),
                "adf_crit_5%":  round(adf_crit["5%"], 4),
                "adf_crit_10%": round(adf_crit["10%"], 4),
                "hedge_ratio":  round(hedge_ratio, 4),
            })

        if not rows:
            return pd.DataFrame(columns=[
                "x", "y", "npd",
                "eg_stat", "eg_pvalue",
                "adf_stat", "adf_pvalue", "adf_crit_1%", "adf_crit_5%", "adf_crit_10%",
                "half_life", "hurst", "hedge_ratio",
            ])

        return pd.DataFrame(rows).sort_values("eg_pvalue").reset_index(drop=True)
    
# ── helpers ──────────────────────────────────────────────────────────────────

def _npd(x: pd.Series, y: pd.Series) -> float:
    """Normalized Price Distance: RMS of the difference of two rebased series."""
    nx = x / x.iloc[0]
    ny = y / y.iloc[0]
    return float(np.sqrt(((nx - ny) ** 2).mean()))


def _ols_residuals(x: pd.Series, y: pd.Series) -> tuple[float, pd.Series]:
    """OLS  y ~ x + const.  Returns (hedge_ratio, residuals)."""
    X = np.column_stack([x.values, np.ones(len(x))])
    fit = OLS(y.values, X).fit()
    return float(fit.params[0]), pd.Series(fit.resid, index=x.index)


def _hurst(spread: pd.Series) -> float:
    """
    Hurst exponent via variance-of-lags method.

    Fits  log Var(τ) ~ 2H · log(τ)  over a range of lags τ.

    H < 0.5  → mean-reverting   (good for pairs trading)
    H = 0.5  → random walk
    H > 0.5  → trending
    """
    s = spread.dropna().values
    n = len(s)
    max_lag = min(n // 4, 100)
    lags = np.unique(np.logspace(1, np.log10(max_lag), num=20).astype(int))
    lags = lags[lags >= 2]

    variances = np.array([np.var(s[lag:] - s[:-lag]) for lag in lags])
    mask = variances > 0
    if mask.sum() < 2:
        return float("nan")

    log_lags = np.log(lags[mask])
    log_vars = np.log(variances[mask])
    slope = float(np.polyfit(log_lags, log_vars, 1)[0])
    return slope / 2.0


def _half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion via AR(1): Δs_t = λ·s_{t-1} + ε."""
    delta = spread.diff().dropna()
    lag = spread.shift(1).dropna()
    delta, lag = delta.align(lag, join="inner")
    X = np.column_stack([lag.values, np.ones(len(lag))])
    lam = float(OLS(delta.values, X).fit().params[0])
    if lam >= 0:
        return float("nan")
    return float(-np.log(2) / np.log(1 + lam))


# ── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    import yfinance as yf
    warnings.filterwarnings("ignore")


    PERIOD   = "730d"
    INTERVAL = "1h"

    from spreadpy.data import get_all_tickers, load_futures_universe
    TICKERS  = get_all_tickers()
    universe = load_futures_universe()
    flat_names = {t: name for cat in universe.values() for t, name in cat.items()}

    print(f"Downloading {len(TICKERS)} futures ({PERIOD} daily)...")
    raw = yf.download(TICKERS, period=PERIOD, interval=INTERVAL,
                      progress=False, auto_adjust=True)["Close"]

    series: list[PriceTimeSeries] = []
    from tqdm import tqdm
    for ticker in tqdm(TICKERS):
        if ticker not in raw.columns:
            print(f"  [skip] {ticker} — not in downloaded data")
            continue
        col = raw[ticker].dropna()
        if len(col) < 60:
            print(f"  [skip] {ticker} — not enough data ({len(col)} bars)")
            continue
        series.append(PriceTimeSeries(col, name=ticker))

    print(f"{len(series)} series loaded. Scanning pairs...\n")

    finder = PairFinder(series, significance=0.05, npd_threshold=None, log_prices=True)
    results = finder.scan()

    if results.empty:
        print("No cointegrated pairs found.")
    else:
        results.insert(2, "x_name", results["x"].map(flat_names))
        results.insert(3, "y_name", results["y"].map(flat_names))

        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", 120)
        pd.set_option("display.float_format", "{:.4f}".format)

        print(f"Found {len(results)} cointegrated pair(s):\n")
        print(results.to_string(index=True))
        results.to_csv("log-pair.csv", index=False)