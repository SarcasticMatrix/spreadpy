from __future__ import annotations


from itertools import combinations
from tqdm import tqdm

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller, coint

from spreadpy.data import PriceTimeSeries




class PairFinder:
    """Scan a universe of assets and rank candidate cointegrated pairs.

    For each pair (s₁, s₂) the following sequential pipeline is applied:

    1. **NPD pre-filter** — normalised price distance. Skips pairs whose
       rebased price paths diverge too much, avoiding expensive statistical
       tests on clearly non-cointegrated pairs.
    2. **Engle-Granger cointegration test** — bivariate OLS + ADF on the
       OLS residuals. Reports the EG p-value.
    3. **ADF on OLS residuals** — direct stationarity test on the spread.
    4. **Half-life** — mean-reversion speed via AR(1) on the spread.
    5. **Hurst exponent** — degree of mean-reversion from the log-variance method.

    Only pairs passing both the EG and ADF significance thresholds are returned.

    :param list[PriceTimeSeries] series: Universe of price series to scan.
    :param float significance: p-value threshold applied to both the EG and
        ADF tests (default 0.05).
    :param float npd_threshold: Maximum normalised price distance to pass the
        pre-filter. Set to ``None`` to disable the pre-filter.
    :param bool log_prices: If ``True``, apply log-transform to all series
        before running any test. Recommended when prices span very different
        scales or when using the Kalman filter downstream.
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
        """Scan all pairs in the universe and return a ranked DataFrame.

        Iterates over all C(n, 2) pairs, applies the NPD pre-filter, runs
        the Engle-Granger and ADF tests, and retains only pairs that pass
        both significance thresholds. Results are sorted by EG p-value
        ascending (strongest cointegration first).

        :returns: DataFrame with one row per qualifying pair, columns:
            ``x``, ``y``, ``npd``, ``eg_stat``, ``eg_pvalue``,
            ``adf_stat``, ``adf_pvalue``, ``adf_crit_1%``, ``adf_crit_5%``,
            ``adf_crit_10%``, ``half_life``, ``hurst``, ``hedge_ratio``.
            Returns an empty DataFrame (with the same columns) if no pair
            passes both thresholds.
        :rtype: pd.DataFrame
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
    """Compute the Normalised Price Distance (NPD) between two series.

    Both series are rebased to 1 at t = 0, then the NPD is:

        NPD = √( (1/T) · Σ_{t=1}^{T} (x_t/x_0 − y_t/y_0)² )

    A small NPD indicates that the two series track each other closely
    in relative terms.

    :param pd.Series x: First price series.
    :param pd.Series y: Second price series.

    :returns: Normalised price distance (non-negative).
    :rtype: float
    """
    nx = x / x.iloc[0]
    ny = y / y.iloc[0]
    return float(np.sqrt(((nx - ny) ** 2).mean()))


def _ols_residuals(x: pd.Series, y: pd.Series) -> tuple[float, pd.Series]:
    """Fit OLS regression y ~ β·x + α and return residuals.

    Solves the normal equations for:

        [β, α]^T = argmin ||y − [x | 1] · [β, α]^T||²

    :param pd.Series x: Independent leg (regressor), shape (T,).
    :param pd.Series y: Dependent leg (regressand), shape (T,).

    :returns: ``(hedge_ratio, residuals)`` — the OLS slope β and the
        in-sample residuals ε_t = y_t − β·x_t − α, indexed as ``x``.
    :rtype: tuple[float, pd.Series]
    """
    X = np.column_stack([x.values, np.ones(len(x))])
    fit = OLS(y.values, X).fit()
    return float(fit.params[0]), pd.Series(fit.resid, index=x.index)


def _hurst(spread: pd.Series) -> float:
    """Estimate the Hurst exponent via the variance-of-lags method.

    For a range of lags τ, the empirical variance of lag-τ differences is:

        Var(τ) = Var(s_t − s_{t−τ})

    The Hurst exponent H is estimated by OLS on:

        log Var(τ) ≈ 2H · log(τ)  ⟹  H = slope / 2

    Interpretation:

        H < 0.5 — mean-reverting (sub-diffusive)
        H = 0.5 — random walk (Brownian motion)
        H > 0.5 — trending (super-diffusive)

    :param pd.Series spread: Spread residual series.

    :returns: Hurst exponent estimate. Returns NaN if fewer than two valid
        lag variances are available.
    :rtype: float
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
    """Estimate the mean-reversion half-life of a spread series via AR(1).

    Fits the discrete-time Ornstein-Uhlenbeck regression:

        Δs_t = λ · s_{t−1} + α + ε_t,   ε_t ~ N(0, σ²)

    The half-life is the time (in bars) for a deviation from equilibrium to
    decay by half:

        τ_{1/2} = −log 2 / log(1 + λ)

    :param pd.Series spread: Spread residual series.

    :returns: Half-life in bars. Returns NaN if λ ≥ 0 (non-mean-reverting).
    :rtype: float
    """
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