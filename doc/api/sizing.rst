######
Sizing
######

Position sizing models that convert a :class:`~spreadpy.signal.signal.Signal`
into absolute quantities for each leg.

PositionSizer
=============

Abstract base class for all position sizers.

.. autoclass:: spreadpy.sizing.positionSizer.PositionSizer
    :members:

LinearSizer
===========

.. autoclass:: spreadpy.sizing.sizers.linearSizer.LinearSizer
    :members:

InverseVolSizer
===============

.. autoclass:: spreadpy.sizing.sizers.inverseVolSizer.InverseVolSizer
    :members:

Kelly sizers
============

Three variants of the second-order Kelly criterion derived from truncated
normal distributions of the entry and exit z-scores.

KellyTruncatedEntry
-------------------

.. autoclass:: spreadpy.sizing.sizers.kellySizers.KellyTruncatedEntry
    :members:

KellyTruncatedExit
------------------

.. autoclass:: spreadpy.sizing.sizers.kellySizers.KellyTruncatedExit
    :members:

KellyTruncatedBoth
------------------

.. autoclass:: spreadpy.sizing.sizers.kellySizers.KellyTruncatedBoth
    :members:
