# Position sizing in pairs trading

```{admonition} Topics
:class: seealso

Position sizing · Kelly criterion · inverse volatility · truncated normal · mean-reversion ·
modern portfolio optimization
```

Generating a signal tells you *when* and *in which direction* to trade. Position sizing
answers the equally important question: *how much?* A strategy with a genuine edge can still
lose money if positions are systematically too large (blow-ups from adverse moves) or too
small (alpha is earned but never extracted meaningfully). Sizing is where portfolio theory
meets execution.

This note develops the theoretical foundations — from Markowitz to Kelly — and then shows
how `spreadpy` translates each into a concrete sizer, with comparative results.

---

## The pairs trade as a single bet

When the z-score $z_t = (s_t - \mu_t)/\sigma_t$ of the spread $s_t = y_t - \beta_t x_t$
crosses the entry threshold $z_{\text{entry}}$, we go short the spread:

- sell $q_y$ units of $y$ at price $p_y$
- buy $q_x = |\beta_t| \cdot q_y \cdot p_y / p_x$ units of $x$ at price $p_x$

The two legs are linked through the hedge ratio $\beta_t$ so that, to first order, the
P&L is driven only by the spread, not by the direction of each individual asset.

The **notional** of the y-leg is $N_y = q_y \cdot p_y$. The x-leg notional is
$N_x = q_x \cdot p_x = |\beta_t| \cdot N_y$. The sizing problem reduces to choosing
$N_y$, or equivalently the **capital fraction**

$$
f_t = \frac{N_y}{\text{capital}_t}.
$$

The gain per unit of spread-sigma when the position closes at $z_{\text{revert}}$ is:

$$
G = z_t - z_{\text{revert}},
$$

where $z_t$ is the z-score at entry and $z_{\text{revert}}$ the z-score at exit.

---

## Markowitz: target-volatility sizing

### From mean-variance to inverse volatility

Modern Portfolio Theory (Markowitz, 1952) seeks the portfolio that minimises variance for
a given expected return. In the single-asset case, the optimal allocation is proportional
to the Sharpe ratio:

$$
f^{MV} = \frac{\mu}{\gamma \sigma^2},
$$

where $\mu$ is the expected return, $\sigma^2$ the variance, and $\gamma > 0$ the investor's
risk-aversion coefficient. The problem is that $\mu$ and $\gamma$ are hard to estimate
reliably. A pragmatic simplification — **target-volatility** sizing — fixes the risk budget
instead:

> Allocate as much capital as needed so that a 1-$\sigma$ adverse move of the spread costs
> exactly a target fraction $\sigma^*$ of current capital.

If the spread residual has rolling standard deviation $\sigma_t$ (estimated over the last
$w$ bars), the resulting capital fraction is:

$$
f_t = \min\!\left(\frac{\sigma^*}{\sigma_t},\, f_{\max}\right),
$$

where $f_{\max}$ is a hard cap. When the spread is calm ($\sigma_t$ small), the sizer
increases the position; when the spread is volatile, it reduces it — a built-in
**volatility targeting** mechanism.

```{admonition} Link to the z-score signal
:class: note

The rolling $\sigma_t$ used here is computed over the same window as the z-score in
`ZScoreSignal`, so the two components are consistent: a position sized for a 1-$\sigma$
adverse move is exactly what the signal threshold parameterises.
```

### `InverseVolSizer` in spreadpy

```python
from spreadpy.sizing.sizers.inverseVolSizer import InverseVolSizer

sizer = InverseVolSizer(
    window=60,          # rolling window for σ_t (bars)
    target_vol=0.02,    # 2% of capital at risk for a 1-σ move
    f_max=0.5,          # never allocate more than 50% of capital
)

# fit must be called first to precompute the rolling σ_t
sizer.fit(spread)

qty_y, qty_x = sizer.size(signal, price_y, price_x, hedge_ratio, capital)
```

The fraction $f_t$ varies bar by bar. The y-leg notional is $N_y = f_t \cdot \text{capital}_t$,
and the x-leg quantity is automatically hedged via $\beta_t$.

---

## The Kelly criterion

### Maximising expected log-wealth

The Kelly criterion (Kelly, 1956) answers the question: *what fraction $f$ of capital
maximises the long-run growth rate of wealth?* It derives from maximising the expected
log-utility of terminal wealth:

$$
f^* = \operatorname{argmax}_f \; \mathbb{E}[\log(1 + f G)],
$$

where $G$ is the random gain per dollar risked. Under Gaussian returns, this gives the
celebrated result:

$$
f^* = \frac{\mathbb{E}[G]}{\mathrm{Var}(G)},
$$

which is proportional to the Sharpe ratio of the bet. Full Kelly maximises growth but is
extremely aggressive — a single bad run can permanently destroy a large fraction of capital.
In practice, **half-Kelly** ($f = f^*/2$) is more common.

### Second-order approximation

For small gains, $\log(1 + fG) \approx fG - \frac{1}{2}f^2G^2$, which gives:

$$
\mathbb{E}[\log(1 + fG)] \approx f\,\mathbb{E}[G] - \frac{f^2}{2}\,\mathbb{E}[G^2].
$$

Maximising over $f$ (first-order condition) yields the **second-order Kelly fraction**:

$$
f^* \approx \frac{\mathbb{E}[G]}{\mathbb{E}[G^2]} = \frac{\mathbb{E}[G]}{\mathrm{Var}(G) + \mathbb{E}[G]^2},
$$

which is well-defined even when $G$ has a non-Gaussian distribution, and easier to estimate
than the exact optimum.

### Kelly for mean-reversion: the truncated normal model

In pairs trading the spread z-score is approximately standard normal. We introduce two
fixed **threshold parameters**:

- $z_e > 0$ — the entry threshold: a position opens when $|z_t| \geq z_e$
- $z_r \in [0, z_e)$ — the reversion target: the position closes when $|z_t|$ returns to $z_r$

The actual z-score values **at the moment** of entry and exit are random variables — we
do not know in advance at exactly which bar the signal will fire or when the spread will
revert. Denote:

- $z$ — the realized entry z-score (random, $z \geq z_e$)
- $\tilde{z}$ — the realized exit z-score (random, $\tilde{z} \leq z_r$)

The per-trade gain per unit of spread-sigma is then:

$$
G = z - \tilde{z}.
$$

Modelling the distributions of $z$ and $\tilde{z}$ as truncated normals lets us compute
$\mathbb{E}[G]$ and $\mathrm{Var}(G)$ in closed form.

**Key tool — Inverse Mills ratios.** For a truncated standard normal, the conditional mean
involves the ratio of the PDF $\varphi$ to the CDF $\Phi$. Define:

$$
\lambda_+(a) = \frac{\varphi(a)}{1 - \Phi(a)}, \qquad \lambda_-(b) = \frac{\varphi(b)}{\Phi(b)},
$$

where $\varphi$ and $\Phi$ are the standard normal PDF and CDF, $\lambda_+(a)$ is the
left-truncation inverse Mills ratio (for a variable conditioned on being $\geq a$), and
$\lambda_-(b)$ is the right-truncation ratio (conditioned on being $\leq b$). Both
are positive and grow as the truncation point moves into the tail.

---

## The three Kelly sizers in `spreadpy`

The three sizers differ in which of the two unknowns — entry level $z$ or exit level
$\tilde{z}$ — is treated as random.

### `KellyTruncatedEntry` — random entry, deterministic exit

A trade opens at bar $t$ only when $z_t \geq z_e$. Conditioned on this event, the entry
z-score is a left-truncated standard normal. Setting
$\lambda_+ = \lambda_+(z_e) = \varphi(z_e)/(1-\Phi(z_e))$, the standard result for
truncated normals gives:

$$
\mathbb{E}[z_t \mid z_t \geq z_e] = \lambda_+, \qquad
\mathrm{Var}(z_t \mid z_t \geq z_e) = 1 - \lambda_+(\lambda_+ - z_e).
$$

The exit $z_r$ is treated as a deterministic constant. The per-trade gain
$G = z_t - z_r$, conditioned on a trade having opened, has moments:

$$
\mathbb{E}[G \mid z_t \geq z_e] = \lambda_+ - z_r, \qquad
\mathrm{Var}(G \mid z_t \geq z_e) = 1 - \lambda_+(\lambda_+ - z_e).
$$

The Kelly objective is maximised conditionally on the trade being active, giving:

$$
f^* = \frac{\lambda_+ - z_r}{1 - \lambda_+(\lambda_+ - z_e) + (\lambda_+ - z_r)^2}.
$$

This fraction is **constant** — computed once at construction from $z_e$ and $z_r$.

### `KellyTruncatedExit` — deterministic entry, random exit

The position closes when the spread reverts below $z_r$. Conditioned on this event, the
exit z-score is a right-truncated standard normal. Setting
$\lambda_- = \lambda_-(z_r) = \varphi(z_r)/\Phi(z_r)$, the standard result gives:

$$
\mathbb{E}[\tilde{z}_t \mid \tilde{z}_t \leq z_r] = -\lambda_-, \qquad
\mathrm{Var}(\tilde{z}_t \mid \tilde{z}_t \leq z_r) = 1 - \lambda_-(\lambda_- + z_r).
$$

The entry z-score $z_t$ is the value observed when the signal fires — a known constant at
sizing time. The per-trade gain $G = z_t - \tilde{z}_t$, conditioned on the exit event,
has moments:

$$
\mathbb{E}[G \mid \tilde{z}_t \leq z_r] = z_t + \lambda_-, \qquad
\mathrm{Var}(G \mid \tilde{z}_t \leq z_r) = 1 - \lambda_-(\lambda_- + z_r).
$$

The Kelly objective is maximised conditionally on the exit being a reversion below $z_r$:

$$
f^*(z_t) = \frac{z_t + \lambda_-}{1 - \lambda_-(\lambda_- + z_r) + (z_t + \lambda_-)^2}.
$$

This fraction **depends on $z_t$** and is recomputed at each bar: the deeper the observed
entry z-score, the larger $\mathbb{E}[G \mid \tilde{z}_t \leq z_r]$, and the larger the
allocated fraction.

### `KellyTruncatedBoth` — both random, independent

Both the entry and exit z-scores are conditioned on their respective threshold events,
and modelled as independent:

$$
z_t \mid z_t \geq z_e \;\perp\; \tilde{z}_t \mid \tilde{z}_t \leq z_r.
$$

With $\lambda_+ = \lambda_+(z_e)$ and $\lambda_- = \lambda_-(z_r)$, the per-trade gain
$G = z_t - \tilde{z}_t$, conditioned on both the entry and exit events, has moments:

$$
\mathbb{E}[G \mid z_t \geq z_e,\, \tilde{z}_t \leq z_r] = \lambda_+ + \lambda_-,
$$

$$
\begin{align*}
\mathrm{Var}(G \mid z_t \geq z_e,\, \tilde{z}_t \leq z_r)
&= \mathrm{Var}(z_t \mid z_t \geq z_e) + \mathrm{Var}(\tilde{z}_t \mid \tilde{z}_t \leq z_r) \\
&= \bigl[1 - \lambda_+(\lambda_+ - z_e)\bigr] + \bigl[1 - \lambda_-(\lambda_- + z_r)\bigr],
\end{align*}
$$

where independence allows the conditional variances to add. The Kelly fraction is:

$$
f^* = \frac{\lambda_+ + \lambda_-}{2 - \lambda_+(\lambda_+ - z_e) - \lambda_-(\lambda_- + z_r) + (\lambda_+ + \lambda_-)^2}.
$$

Like `KellyTruncatedEntry`, this fraction is **constant** — both conditioning events are
fully characterised by the fixed thresholds $z_e$ and $z_r$.

---

## How $f^*$ behaves as a function of $z_e$

Setting $z_r = 0$ (exit at the mean), $\lambda_- = \lambda_-(0) = \varphi(0)/\Phi(0) \approx 0.80$
(fixed), and varying $z_e$:

| $z_e$ | $\lambda_+(z_e)$ | $f^*$ (Entry, uncapped) | $f^*$ (Both, uncapped) |
|:---:|:---:|:---:|:---:|
| 0.5 | 1.14 | 0.73 | 0.44 |
| 1.0 | 1.53 | 0.60 | 0.39 |
| 1.5 | 1.94 | 0.50 | 0.34 |
| 2.0 | 2.37 | 0.41 | 0.30 |
| 2.5 | 2.84 | 0.35 | 0.27 |

Values computed from the closed-form formulas; a hard cap $f_{\max} = 0.5$ would clip the
first two rows of the Entry column. The $\lambda_-$ term in the Both column
pulls $f^*$ down relative to Entry, because the randomness in the exit adds variance to
$G$ without changing $\mathbb{E}[G]$ proportionally.

As $z_e$ increases, $\lambda_+(z_e)$ grows (the conditional mean of the entry z-score is
higher), but the variance of the truncated distribution shrinks rapidly, and $\mathbb{E}[G]^2$
dominates the denominator $\mathbb{E}[G^2]$. The net effect is that $f^*$ **decreases**
with $z_e$. Counterintuitively, **waiting for a very large z-score does not justify a
larger position** — the distribution becomes more concentrated but the probability of
occurrence drops exponentially, and the optimal fraction shrinks accordingly.

---

## Comparing the sizers in `spreadpy`

### Setup

```python
import numpy as np
import yfinance as yf
from spreadpy.data import PriceTimeSeries
from spreadpy.spread.hedgeRatio.kalmanFilter import KalmanFilter
from spreadpy.spread import SpreadSeries
from spreadpy.signal.zScoreSignal import ZScoreSignal
from spreadpy.sizing.sizers.linearSizer import LinearSizer
from spreadpy.sizing.sizers.inverseVolSizer import InverseVolSizer
from spreadpy.sizing.sizers.kellySizers import (
    KellyTruncatedEntry, KellyTruncatedExit, KellyTruncatedBoth
)
from spreadpy.backtest.engine import BacktestEngine

raw = yf.download(["GLD", "GDX"], start="2018-01-01", end="2024-01-01",
                  auto_adjust=True)["Close"].dropna()
y = PriceTimeSeries(raw["GLD"], name="GLD")
x = PriceTimeSeries(raw["GDX"], name="GDX")
```

### One signal, four sizers

```python
estimator  = KalmanFilter(alpha=1e-5)
signal_gen = ZScoreSignal(window=60, entry_threshold=1.0, revert_threshold=0.0)

sizers = {
    "Linear":       LinearSizer(max_notional=50_000),
    "Inverse-Vol":  InverseVolSizer(window=60, target_vol=0.02, f_max=0.5),
    "Kelly-Entry":  KellyTruncatedEntry(z_entry=1.0, z_revert=0.0, f_max=0.5),
    "Kelly-Exit":   KellyTruncatedExit(z_revert=0.0, f_max=0.5),
    "Kelly-Both":   KellyTruncatedBoth(z_entry=1.0, z_revert=0.0, f_max=0.5),
}

results = {}
for name, sizer in sizers.items():
    engine = BacktestEngine(
        estimator=estimator,
        signal_gen=signal_gen,
        sizer=sizer,
        initial_capital=1_000_000,
        train_frac=0.6,
        val_frac=0.2,
        log_prices=True,
    )
    _, test = engine.run(y, x)
    results[name] = test.metrics
```

### Reading the results

```python
import pandas as pd
summary = pd.DataFrame(results).T[
    ["annualised_return", "volatility", "sharpe", "max_drawdown", "sortino"]
]
print(summary.round(3))
```

Expected output (illustrative — actual results depend on the period):

| Sizer | Ann. Return | Volatility | Sharpe | Max DD | Sortino |
|---|---:|---:|---:|---:|---:|
| Linear | +6.1% | 8.4% | 0.73 | −9.2% | 1.02 |
| Inverse-Vol | +7.8% | 7.1% | 1.10 | −7.4% | 1.58 |
| Kelly-Entry | +8.3% | 9.2% | 0.90 | −10.1% | 1.31 |
| Kelly-Exit | +9.7% | 10.8% | 0.90 | −12.3% | 1.29 |
| Kelly-Both | +8.6% | 9.5% | 0.91 | −10.8% | 1.33 |

### What these numbers tell us

**Linear sizer** is the weakest: it sizes by z-score magnitude, ignoring both volatility
and the probabilistic structure of the trade. It earns alpha but leaves return on the
table.

**Inverse-vol** achieves the best risk-adjusted metrics (Sharpe, Sortino, drawdown) because
it explicitly targets a volatility budget. It is the right default for a risk-constrained
mandate.

**Kelly sizers** maximise expected log-growth, not the Sharpe ratio — so they tend to be
more aggressive. `KellyTruncatedExit` is the most dynamic: the position size scales
directly with the observed $z_t$ at entry, concentrating capital at the most favourable
opportunities. This boosts raw returns at the cost of higher drawdown.

```{admonition} Which sizer to choose?
:class: tip

- **Risk-constrained mandate** (target vol, regulatory capital): `InverseVolSizer`
- **Growth-oriented mandate** (maximise CAGR): `KellyTruncatedBoth` with $f_{\max} = 0.3$
- **Dynamic sizing** (position scales with signal strength): `KellyTruncatedExit`
- **Baseline / sanity check**: `LinearSizer`

In all cases, cap the fraction at $f_{\max} \leq 0.5$ and combine with a walk-forward
validation to ensure the assumed distributions ($z_{\text{entry}}$, $z_{\text{revert}}$)
match the actual trading behaviour.
```

---

## References

- Kelly, J. L. (1956). *A new interpretation of information rate.* Bell System Technical
  Journal, 35(4), 917–926.
- Markowitz, H. (1952). *Portfolio selection.* Journal of Finance, 7(1), 77–91.
- MacLean, L. C., Thorp, E. O. & Ziemba, W. T. (2010). *The Kelly Capital Growth
  Investment Criterion.* World Scientific.
- Palomar, D. P. (2024). *Portfolio Optimization: Theory and Application.* Ch. 15.
