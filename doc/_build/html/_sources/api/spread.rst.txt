######
Spread
######

Hedge ratio estimators and spread series diagnostics.

SpreadSeries
============

.. autoclass:: spreadpy.spread.spreadSeries.SpreadSeries
    :members:

HedgeRatioEstimator
===================

Abstract base class for all hedge ratio estimators.

.. autoclass:: spreadpy.spread.hedgeRatioEstimator.HedgeRatioEstimator
    :members:

ConstantOLS
===========

.. autoclass:: spreadpy.spread.hedgeRatio.constantOLS.ConstantOLS
    :members:

RollingOLS
==========

.. autoclass:: spreadpy.spread.hedgeRatio.rollingOLS.RollingOLS
    :members:

KalmanFilter
============

.. autoclass:: spreadpy.spread.hedgeRatio.kalmanFilter.KalmanFilterParams
    :members:

.. autoclass:: spreadpy.spread.hedgeRatio.kalmanFilter.KalmanFilter
    :members:

KalmanFilterWithVelocity
========================

.. autoclass:: spreadpy.spread.hedgeRatio.kalmanFilterWithVelocity.KalmanFilterWithVelocityParams
    :members:

.. autoclass:: spreadpy.spread.hedgeRatio.kalmanFilterWithVelocity.KalmanFilterWithVelocity
    :members:
