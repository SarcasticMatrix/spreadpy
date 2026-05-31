~~~~~~~~~~~~
Explorations
~~~~~~~~~~~~

A collection of mathematical notes exploring questions that naturally arise when using ``hestonpy``.
Each post mixes theory, intuition, and concrete code to dig into one specific question.

.. grid:: 1
   :gutter: 4

   .. grid-item-card:: Tracking the hedge ratio with the Kalman filter
      :link: kalman_filter
      :link-type: doc
      :shadow: md

      The hedge ratio between two cointegrated assets is rarely constant. We derive the
      Kalman filter state-space model used by ``spreadpy`` — both the 2-state random-walk
      and the 3-state locally-linear trend variants — and show how each avoids lookahead
      bias. Applied to GLD/GDX with a full stationarity diagnostic.

      +++
      **Topics:** Kalman filter · state-space model · dynamic hedge ratio · lookahead bias

   .. grid-item-card:: Validating a pairs trading strategy
      :link: strategy_validation
      :link-type: doc
      :shadow: md

      A backtest is an optimistic experiment by construction. We identify the four failure
      modes of naive backtesting — single split, multiple testing, autocorrelation bias,
      period overfitting — and design a ``StrategyValidator`` class with walk-forward
      engine, Deflated Sharpe Ratio, PBO, and parameter sensitivity analysis.

      +++
      **Topics:** walk-forward · DSR · PBO · autocorrelation · StrategyValidator


.. toctree::
   :hidden:
   :maxdepth: 1

   arbitrage_free
   kalman_filter
   strategy_validation
