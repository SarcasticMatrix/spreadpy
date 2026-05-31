# Validating a pairs trading strategy

```{admonition} Topics
:class: seealso

Walk-forward · overfitting · Deflated Sharpe Ratio · Probability of Backtest Overfitting ·
Minimum Track Record Length · autocorrelation correction · parameter sensitivity
```

```{admonition} Status
:class: note

This is a **design document** — a reflection on what needs to be built in `spreadpy`
to properly validate a strategy. Nothing described here is implemented yet.
```

A backtest is an optimistic experiment by construction: we fit a model on historical data,
evaluate it on the same (or adjacent) data, and report the result. Every choice — the
pair, the estimator, the signal threshold, the sizer — is at least partially informed by
the data we are testing on. The result is **selection bias**: we over-report performance
that will not repeat out of sample.

This note identifies the four failure modes of naive backtesting and proposes a concrete
validation stack for `spreadpy`, centred on a `StrategyValidator` class.

---

## The four failure modes

### 1. A single train/test split is not enough

The current `BacktestEngine` splits data once into train / val / test. This is a solid
baseline, but a strategy that performs well on *one* test period may simply be lucky.
A genuine edge should survive across *many* consecutive out-of-sample windows.

**What to build:** `WalkForwardEngine` — rolls the train window forward in fixed steps,
re-fits the full pipeline at each fold, and aggregates OOS results.

### 2. Multiple testing inflates the Sharpe ratio

Suppose we run $N$ parameter configurations and keep the best. Let $r_1, \ldots, r_T$
be the bar-level returns of a single configuration, with $T$ the number of bars, sample
mean $\bar{r}$, and sample standard deviation $\hat{\sigma}$. The per-bar sample Sharpe
ratio is $\widehat{SR} = \bar{r}/\hat{\sigma}$.

Under the null hypothesis of zero true edge, $\widehat{SR}_k$ is approximately normally
distributed for each configuration $k$, with standard deviation:

$$
\sigma_{\widehat{SR}} = \sqrt{\frac{1}{T}\left(1 + \frac{\widehat{SR}_k^2}{2}\right)} \approx \frac{1}{\sqrt{T}},
$$

where the approximation holds when $\widehat{SR}_k \approx 0$. The expected maximum of $N$
such independent statistics is then:

$$
\mathbb{E}\!\left[\max_{k=1}^N \widehat{SR}_k\right] \approx \sigma_{\widehat{SR}} \cdot \Phi^{-1}\!\left(1 - \frac{1}{N}\right),
$$

where $\Phi^{-1}$ is the standard normal quantile function. Even for $N = 20$ and
$T = 252$ bars, this expectation is $\approx 0.12$ per bar, equivalent to an annualised
Sharpe of $0.12\sqrt{252} \approx 1.9$ from pure noise.

**What to build:** `DeflatedSharpeRatio` — penalises the observed Sharpe for the number
of trials and the non-normality of returns.

### 3. Returns are autocorrelated — the Sharpe ratio is biased

The standard annualised Sharpe $\widehat{SR}_\text{ann} = \widehat{SR} \cdot \sqrt{n}$,
where $n$ is the number of bars per year (e.g. $n = 252$ for daily data), assumes i.i.d.
returns. Mean-reversion strategies produce **positively autocorrelated** returns at short
horizons: a winning day is followed by another winning day while the open position
continues to revert. Positive serial correlation inflates $\widehat{SR}_\text{ann}$ because
the i.i.d. formula underestimates the true variance of multi-period returns.

Lo (2002) derives the correction. Let $\hat{\rho}_k$ be the sample autocorrelation at lag
$k$, defined as:

$$
\hat{\rho}_k = \frac{\sum_{t=k+1}^T (r_t - \bar{r})(r_{t-k} - \bar{r})}{\sum_{t=1}^T (r_t - \bar{r})^2},
$$

and let $q$ be the truncation lag (e.g. chosen by Newey-West). The long-run variance
scaling factor is:

$$
\eta = 1 + 2\sum_{k=1}^{q} \left(1 - \frac{k}{q+1}\right)\hat{\rho}_k,
$$

where the Bartlett weights $(1 - k/(q+1))$ downweight higher lags. The
autocorrelation-adjusted annualised Sharpe is then:

$$
\widehat{SR}_\text{adj} = \frac{\widehat{SR}_\text{ann}}{\sqrt{\eta}}.
$$

When $\hat{\rho}_k > 0$ for all $k$, we have $\eta > 1$ and therefore
$\widehat{SR}_\text{adj} < \widehat{SR}_\text{ann}$: the i.i.d. formula overstates
performance. For typical mean-reversion strategies, $\eta \in [1.2, 2.0]$, yielding a
downward correction of 10–40%.

**What to build:** `autocorr_adjusted_sharpe(returns, max_lags)` — applies Lo's correction
before any significance test.

### 4. Overfitting to a specific period

A strategy can be structurally sound but over-tuned to the volatility regime, liquidity
conditions, or correlation structure of the backtest period. This is the hardest failure
mode to detect because the OOS period may look similar to IS on the surface.

**What to build:** `ProbabilityOfBacktestOverfitting` — the combinatorial test of Bailey
& López de Prado (2014).

---

## Proposed building blocks

### `WalkForwardEngine`

Rolls the train+test window forward in steps of `step` bars. At each fold $k$, denoting
by $t_k$ the fold start date, $T_\text{train}$ the training window length, and
$T_\text{test}$ the evaluation window length:

1. Train on $[t_k,\; t_k + T_\text{train})$
2. Evaluate on $[t_k + T_\text{train},\; t_k + T_\text{train} + T_\text{test})$
3. Store `BacktestResult`

```text
WalkForwardEngine(
    estimator, signal_gen, sizer,
    train_size   : int,      # bars in each training window
    test_size    : int,      # bars in each OOS window
    step         : int,      # how many bars to advance each fold
    expanding    : bool,     # True → expanding window, False → rolling
)
.run(y, x) → List[BacktestResult]
.aggregate()  → pd.DataFrame   # one row per fold: Sharpe, MDD, n_trades, …
.plot()
```

A fold summary that immediately shows whether performance is **consistent** across time or
concentrated in one lucky period is the single most useful diagnostic.

Key design question: **expanding vs rolling window?**

- *Expanding* (anchored start): more data each fold → more stable estimates, but older
  data may be structurally irrelevant.
- *Rolling* (fixed size): represents how the strategy would actually be run live, but
  early folds may be noisy due to small training sets.

Both should be supported.

### `DeflatedSharpeRatio`

Bailey & López de Prado (2014) correct the observed Sharpe for three simultaneous biases:
non-normality of returns, number of trials, and track record length.

**Step 1 — Probabilistic Sharpe Ratio (PSR).**

Let $T$ be the number of OOS bar-level return observations, $\widehat{SR}$ the per-bar
sample Sharpe ratio of the selected strategy, $\hat{\gamma}_3$ the sample skewness of
those returns, and $\hat{\gamma}_4$ their sample *excess* kurtosis. Given a benchmark
per-bar Sharpe $SR^*$, the PSR is the estimated probability that the true Sharpe exceeds
$SR^*$:

$$
\widehat{PSR}(SR^*) = \Phi\!\left(
    \frac{(\widehat{SR} - SR^*)\sqrt{T-1}}
    {\sqrt{1 - \hat{\gamma}_3\, \widehat{SR} + \dfrac{\hat{\gamma}_4 - 1}{4}\, \widehat{SR}^2}}
\right),
$$

where $\Phi$ is the standard normal CDF. The denominator inflates the standard error when
returns are skewed ($\hat{\gamma}_3 \neq 0$) or fat-tailed ($\hat{\gamma}_4 > 0$):
non-Gaussian returns make $\widehat{SR}$ a noisier estimator, so the same observed value
carries less statistical evidence.

**Step 2 — Deflated Sharpe Ratio (DSR).**

Suppose $N$ configurations were tested in total. Let $\widehat{SR}_k$ be the per-bar
Sharpe of configuration $k$, and $\overline{SR} = \frac{1}{N}\sum_{k=1}^N \widehat{SR}_k$
their average. The DSR sets $SR^*$ to the expected maximum Sharpe one would observe under
data-mining alone:

$$
SR^* = \Phi^{-1}\!\left(1 - \frac{1}{N}\right)
\cdot \sqrt{\frac{1 - \hat{\gamma}_3\,\overline{SR} + \dfrac{\hat{\gamma}_4-1}{4}\,\overline{SR}^2}{T - 1}}.
$$

The DSR is then $\widehat{PSR}(SR^*)$: the probability that the selected strategy has
genuine alpha beyond what data-mining would produce. A DSR above 0.95 (i.e. $p < 0.05$
under the null) is the target.

```text
DeflatedSharpeRatio(
    returns      : pd.Series,            # OOS bar-level returns of the selected strategy
    n_trials     : int,                  # N = total number of configurations tested
    all_returns  : List[pd.Series],      # returns of all N trials (to compute SR̄)
)
.psr(sr_star)  → float    # PSR at a given benchmark SR*
.dsr()         → float    # DSR = PSR(SR*) with SR* set by data-mining correction
.summary()     → pd.Series
```

### `MinimumTrackRecordLength`

Before testing for selection bias, a more basic question: *do we have enough OOS data to
conclude that $\widehat{SR} > 0$ is not due to chance?*

Inverting the PSR formula at benchmark $SR^* = 0$, the minimum number of OOS bars $T^*$
required to reject $H_0: \text{true } SR \leq 0$ at Type I error level $\delta$ is:

$$
T^* = 1 + \left(1 - \hat{\gamma}_3\, \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\, \widehat{SR}^2\right)
\cdot \left(\frac{\Phi^{-1}(1 - \delta)}{\widehat{SR}}\right)^2.
$$

Note that $\delta$ here is the significance level, not the Kalman filter noise parameter
used elsewhere in `spreadpy`. For concreteness, consider Gaussian returns
($\hat{\gamma}_3 = \hat{\gamma}_4 = 0$) and $\delta = 0.05$:

- For $\widehat{SR}_\text{ann} = 0.5$ with $n = 252$ daily bars: $\widehat{SR} = 0.5/\sqrt{252} \approx 0.031$, giving $T^* \approx 2{,}730$ bars — about **10.8 years** of daily OOS data.
- For $\widehat{SR}_\text{ann} = 1.0$: $\widehat{SR} \approx 0.063$, giving $T^* \approx 683$ bars — about **2.7 years**.

Negative skewness (common in pairs trading due to occasional blow-ups) or excess kurtosis
increases $T^*$ further by making $\widehat{SR}$ a noisier estimator.

```text
MinimumTrackRecordLength(
    sr_annualised    : float,
    skew             : float = 0.0,    # γ̂₃ of returns
    excess_kurt      : float = 0.0,    # γ̂₄ of returns
    significance     : float = 0.05,   # δ
    periods_per_year : int   = 252,    # n
)
.mtrl()  → int     # T* in bars
```

### `ProbabilityOfBacktestOverfitting`

The PBO test (Bailey & López de Prado, 2014) does not require specifying the number of
trials. It uses **combinatorial symmetric cross-validation** on a matrix of returns.

**Setup.** Let $\mathbf{R} \in \mathbb{R}^{T \times N}$ be the matrix of bar-level returns,
where column $k$ contains the $T$ OOS returns of configuration $k$, and $N$ is the total
number of configurations. Partition the $T$ bars into $S$ non-overlapping sub-periods of
equal length $T/S$, yielding index sets $\mathcal{I}_1, \ldots, \mathcal{I}_S$.

**Combinatorial splits.** For each of the $\binom{S}{S/2}$ ways to partition the $S$
sub-periods into an IS set $\mathcal{J} \subset \{1,\ldots,S\}$ of size $S/2$ and a
complementary OOS set $\mathcal{J}^c$:

1. Compute a scalar performance metric $\mu_k^\text{IS}$ (e.g. the Sharpe ratio) for
   each configuration $k$, using only the rows of $\mathbf{R}$ belonging to the IS
   sub-periods $\bigcup_{j \in \mathcal{J}} \mathcal{I}_j$.
2. Select the best IS configuration $k^* = \arg\max_k\, \mu_k^\text{IS}$.
3. Evaluate $k^*$ on the OOS sub-periods: $\mu_{k^*}^\text{OOS}$.
4. Record its relative OOS rank via the logit-transformed percentile:

   ```{math}
   \omega = \Phi\!\left(
       \frac{\mu_{k^*}^{\text{OOS}} - \bar{\mu}^{\text{OOS}}}{\hat{\sigma}^{\text{OOS}}}
   \right),
   ```

   where $\bar{\mu}^{\text{OOS}}$ and $\hat{\sigma}^{\text{OOS}}$ are the mean and
   standard deviation of $\{\mu_k^{\text{OOS}}\}_{k=1}^N$ across all $N$ configurations
   on that OOS subset.

**PBO.** The probability of backtest overfitting is the fraction of splits where the IS
winner ranks below median OOS ($\omega < 0.5$):

$$
PBO = \frac{1}{\binom{S}{S/2}} \sum_{\text{splits}} \mathbf{1}[\omega < 0.5].
$$

A $PBO > 0.5$ means the IS winner is more likely than not to be below-median OOS —
strong evidence of overfitting. A $PBO \approx 0$ means the best IS configuration
consistently wins OOS too.

```text
ProbabilityOfBacktestOverfitting(
    returns_matrix : pd.DataFrame,   # T × N, column k = returns of configuration k
    n_partitions   : int = 16,       # S
    metric         : str = "sharpe", # the scalar μ used for ranking
)
.fit()   → PBOResult
    .pbo            : float          # probability of overfitting
    .logit_sr       : pd.Series      # distribution of ω across splits
    .plot_logit()
    .plot_rank_correlation()
```

### `ParameterSensitivity`

A strategy is over-fitted if its performance is concentrated at a single point in
parameter space. Robustness requires a **plateau**, not a spike.

```text
ParameterSensitivity(
    engine       : BacktestEngine | WalkForwardEngine,
    param_grid   : dict,   # e.g. {"entry_threshold": [0.5, 1.0, 1.5, 2.0]}
    metric       : str = "sharpe",
)
.run(y, x) → pd.DataFrame    # one row per config
.plot_heatmap()               # 2-param heatmap
.plot_profile()               # 1-param sensitivity curve
.robustness_score() → float  # fraction of configs within X% of optimum
```

---

## The `StrategyValidator` class

All the above building blocks are orchestrated by a single entry point:

```text
StrategyValidator(
    engine           : BacktestEngine,
    walk_forward     : WalkForwardEngine | None,
    param_grid       : dict | None,
    n_trials         : int = 1,
    significance     : float = 0.05,   # δ for MTRL and PSR
    periods_per_year : int = 252,
)
.validate(y, x) → ValidationReport
```

### `ValidationReport`

```text
ValidationReport
├── walk_forward_results   : List[BacktestResult]
├── aggregated_metrics     : pd.DataFrame        # per-fold
├── dsr                    : DeflatedSharpeRatioResult
│      .psr, .dsr, .n_trials, .sr_raw
├── mtrl                   : MinimumTrackRecordLengthResult
│      .required_bars, .available_bars, .sufficient
├── pbo                    : PBOResult | None     # if param_grid provided
│      .pbo, .logit_sr
├── sensitivity            : SensitivityResult | None
│      .heatmap, .robustness_score
└── .summary()             → pd.DataFrame        # one-page verdict
    .plot()                → matplotlib figure   # 4-panel dashboard
```

### The one-page verdict

`summary()` would return something like:

| Check | Value | Threshold | Pass? |
|---|---|---|---|
| OOS Sharpe (raw) | 0.82 | > 0 | ✓ |
| OOS Sharpe (Lo-adjusted) | 0.61 | > 0 | ✓ |
| Deflated Sharpe Ratio (DSR) | 0.44 | > 0 | ✓ |
| PSR (p-value) | 0.031 | < 0.05 | ✓ |
| Track record length | 504 bars | ≥ 387 ($T^*$) | ✓ |
| Probability of overfitting | 0.38 | < 0.50 | ✓ |
| Walk-forward consistency | 7/10 folds profitable | ≥ 6/10 | ✓ |
| Robustness score | 0.71 | > 0.50 | ✓ |

---

## Open design questions

Before implementing, a few choices need to be made:

**1. Where does `StrategyValidator` live?**
A new `spreadpy.validation` sub-package seems natural. It would sit alongside
`spreadpy.backtest` and import from it.

**2. How to handle multiple pairs?**
`PBO` and `DSR` are defined for a single strategy with $N$ parameter configurations.
If the user also selected the *pair* by screening, the pair selection is itself a trial
and should count towards $N$.

**3. Should walk-forward use purged folds?**
In time-series cross-validation, adjacent folds share information through overlapping
positions. A **purge gap** of a few bars between train and test prevents data leakage from
open trades that span the fold boundary. This is especially important for short
holding-period strategies.

**4. What is "one trial" for DSR?**
Each unique combination of (estimator type, signal parameters, sizer parameters) is one
trial. The user should declare $N$ explicitly, or `StrategyValidator` could count the
configs in `param_grid` automatically.

---

## Implementation order

Given the above, a natural build order would be:

1. `WalkForwardEngine` — most impactful, immediately usable, no new maths
2. `autocorr_adjusted_sharpe` — one function, easy win, fixes a systematic bias
3. `MinimumTrackRecordLength` — closed-form formula, tells the user if they have enough data
4. `DeflatedSharpeRatio` — requires knowing $N$, slightly more complex
5. `ParameterSensitivity` — useful for tuning, builds on `WalkForwardEngine`
6. `ProbabilityOfBacktestOverfitting` — most complex, requires many backtests
7. `StrategyValidator` + `ValidationReport` — orchestration layer, last step

---

## References

- Bailey, D. H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality.* Journal of Portfolio Management,
  40(5), 94–107.
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014). *Pseudo-Mathematics
  and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample
  Performance.* Notices of the American Mathematical Society, 61(5), 458–471.
- Lo, A. W. (2002). *The Statistics of Sharpe Ratios.* Financial Analysts Journal, 58(4),
  36–52.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. Ch. 11–12.
