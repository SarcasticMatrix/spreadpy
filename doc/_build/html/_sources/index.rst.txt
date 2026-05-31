:sd_hide_title:
:html_theme.sidebar_secondary.remove:

.. raw:: html

    <!-- CSS overrides on the homepage only -->
    <style>
    .bd-main .bd-content .bd-article-container {
    max-width: 95rem; /* Make homepage a little wider instead of 60em */
    }
    /* Extra top/bottom padding to the sections */
    article.bd-article section {
    padding: 7rem 0 7rem;
    }
    /* Override all h1 headers except for the hidden ones */
    h1:not(.sd-d-none) {
    font-weight: bold;
    font-size: 48px;
    text-align: center;
    margin-bottom: 4rem;
    }
    /* Override all h3 headers that are not in hero */
    h3:not(#hero h3) {
    font-weight: bold;
    text-align: center;
    }
    </style>

spreadpy: A pairs trading framework for systematic strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. raw:: html

    <div style="display: flex;">

        <div style="flex: 2.5">
            <div id="hero-left">  <!-- Start Hero -->
                <h2 style="font-size: 60px; font-weight: bold; margin: 5rem auto 0;">spreadpy</h2>
                <h3 style="font-size: 25px; font-weight: bold; margin-top: 0.6rem; text-align: left; padding-left: 0.6rem;">Pairs trading, from spread to backtest</h3>
                <p><code>spreadpy</code> provides a complete pipeline for <b>pairs trading</b>:</p>
                <ul>
                    <li><b>Pair research</b>, cointegration tests, ADF, half-life, Hurst exponent</li>
                    <li><b>Spread construction</b>, OLS, rolling OLS, Kalman filter</li>
                    <li><b>Signal generation</b>, z-score, copula</li>
                    <li><b>Position sizing</b>, notional, inverse-vol, Kelly criterion</li>
                    <li><b>Walk-forward backtesting</b>, slippage & commission costs, risk metrics</li>
                </ul>

                <div class="homepage-button-container">
                <div class="homepage-button-container-row">
                    <a href="./examples/index.html" class="homepage-button primary-button">Examples</a>
                    <a href="./user_guide/index.html" class="homepage-button secondary-button">Exploration</a>
                </div>
                <div class="homepage-button-container-row">
                    <a href="./api/index.html" class="homepage-button-link">See API Reference →</a>
                </div>
                </div>
            </div>  <!-- End Hero -->
        </div>

        <div style="flex: 3; display: flex; align-items: center">
            <img src="_static/_images/Figure_1.png" alt="Backtest example" style="width: 100%">
        </div>

    </div>



Key features
~~~~~~~~~~~~

**An intuitive and modular Python library for systematic pairs trading. Please, contact me if you have any suggestions!**

The ``spreadpy`` Python package implements a complete **pairs trading** pipeline,
from universe scanning to walk-forward backtesting.

Covered topics by the ``spreadpy`` package :

* **Pair research** — scan a universe of assets and rank cointegrated pairs (Engle-Granger, ADF, half-life, Hurst exponent).
* **Spread construction** — estimate time-varying hedge ratios $\beta_t$: constant OLS, rolling OLS, 2-state and 3-state Kalman filters.
* **Signal generation** — z-score entry/exit rules and copula-based conditional CDF signals (Gaussian, Clayton, Gumbel).
* **Position sizing** — notional-based (linear z-score ramp), inverse-volatility (Markowitz), and Kelly criterion (truncated normal, three variants).
* **Backtesting** — walk-forward engine with train / validation / test split, transaction costs (slippage + commission), and a full risk metric suite (Sharpe, Sortino, Calmar, CDaR, …).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ./api/index.rst
   ./examples/index.rst
   ./user_guide/index.rst
