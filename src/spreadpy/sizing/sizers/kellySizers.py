"""
Kelly-based position sizers for spread mean-reversion trading.

Both classes derive :math:`f^*` from the second-order Kelly criterion:

.. math::

    f^* = \\operatorname{argmax}_f \\mathbb{E}[\\log(1 + f G)]
        \\approx \\frac{\\mathbb{E}[G]}{\\mathbb{E}[G^2]}

where the approximation follows from :math:`\\log(1+x) \\approx x - x^2/2`,
giving:

.. math::

    \\frac{\\mathbb{E}[G]}{\\mathbb{E}[G^2]}
    = \\frac{\\mathbb{E}[G]}{\\mathrm{Var}(G) + \\mathbb{E}[G]^2}

The spread z-score is modelled as
:math:`z_t = (X_t - \\mu_t)/\\sigma_t \\sim \\mathcal{N}(0,1)`.
We trade mean-reversion: short when :math:`z_t \\geq z_{\\text{entry}}`,
target :math:`z_{\\text{revert}} < z_{\\text{entry}}`.

Inverse Mills ratios
--------------------
Left truncation at :math:`a` (:math:`z \\geq a`):

.. math::

    \\lambda_+(a) = \\frac{\\varphi(a)}{1 - \\Phi(a)}

Right truncation at :math:`b` (:math:`z \\leq b`):

.. math::

    \\lambda_-(b) = \\frac{\\varphi(b)}{\\Phi(b)}

where :math:`\\varphi` and :math:`\\Phi` are the standard normal PDF and CDF.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm

from spreadpy.signal.signal import Direction, Signal
from spreadpy.sizing.positionSizer import PositionSizer


# ---------------------------------------------------------------------------
# Utility: inverse Mills ratios
# ---------------------------------------------------------------------------

def _mills_left(a: float) -> float:
    """:math:`\\lambda_+(a) = \\varphi(a)/(1 - \\Phi(a))` — left-truncation at :math:`a`."""
    return float(norm.pdf(a) / (1.0 - norm.cdf(a)))


def _mills_right(b: float) -> float:
    """:math:`\\lambda_-(b) = \\varphi(b)/\\Phi(b)` — right-truncation at :math:`b`."""
    return float(norm.pdf(b) / norm.cdf(b))


# ---------------------------------------------------------------------------
# Shared sizing logic
# ---------------------------------------------------------------------------

def _quantities(
    frac: float,
    capital: float,
    price_y: float,
    price_x: float,
    hedge_ratio: float,
) -> Tuple[float, float]:
    notional_y = capital * frac
    qty_y = notional_y / price_y if price_y > 0.0 else 0.0
    qty_x = (notional_y * abs(hedge_ratio)) / price_x if price_x > 0.0 else 0.0
    return qty_y, qty_x


# ---------------------------------------------------------------------------
# Method 1 — KellyTruncatedEntry
# ---------------------------------------------------------------------------

class KellyTruncatedEntry(PositionSizer):
    """Kelly sizer where only the entry level :math:`z` is random.

    The entry z-score is modelled as a truncated standard normal:

    .. math::

        z \\sim \\mathcal{N}(0,1) \\text{ truncated to } [z_{\\text{entry}}, +\\infty),
        \\quad p(z) = \\frac{\\varphi(z)}{1 - \\Phi(z_{\\text{entry}})}

    The reversion target :math:`z_{\\text{revert}}` is treated as a deterministic
    constant. The per-trade gain is :math:`G = z - z_{\\text{revert}}`.

    Moments (:math:`\\lambda_+ = \\lambda_+(z_{\\text{entry}})`):

    .. math::

        \\mathbb{E}[G]   &= \\lambda_+ - z_{\\text{revert}} \\\\
        \\mathrm{Var}(z) &= 1 - \\lambda_+(\\lambda_+ - z_{\\text{entry}}) \\\\
        \\mathbb{E}[G^2] &= \\mathrm{Var}(z) + \\mathbb{E}[G]^2

    Second-order Kelly fraction:

    .. math::

        f^* = \\frac{\\lambda_+ - z_{\\text{revert}}}
               {1 - \\lambda_+(\\lambda_+ - z_{\\text{entry}}) + (\\lambda_+ - z_{\\text{revert}})^2}

    The fraction is constant and computed once at construction.

    :param float z_entry: Entry threshold; positions are opened when :math:`|z_t| \\geq z_{\\text{entry}}`.
    :param float z_revert: Deterministic reversion target (default 0.0, i.e. the mean).
    :param float f_max: Hard cap on the Kelly fraction (default 0.5).
    """

    def __init__(
        self,
        z_entry: float,
        z_revert: float = 0.0,
        f_max: float = 0.5,
    ) -> None:
        self.z_entry  = z_entry
        self.z_revert = z_revert
        self.f_max    = f_max
        self._frac    = self._kelly()

    def _kelly(self) -> float:
        lam_plus = _mills_left(self.z_entry)
        mu_g     = lam_plus - self.z_revert
        if mu_g <= 0.0:
            return 0.0
        var_z = 1.0 - lam_plus * (lam_plus - self.z_entry)
        e_g2  = var_z + mu_g ** 2
        if e_g2 <= 0.0:
            return 0.0
        return float(min(mu_g / e_g2, self.f_max))

    def size(
        self,
        signal: Signal,
        price_y: float,
        price_x: float,
        hedge_ratio: float,
        capital: float = 0.0,
    ) -> Tuple[float, float]:
        """Compute quantities using the constant Kelly fraction f*.

        Returns ``(0, 0)`` when ``signal.direction`` is ``FLAT``,
        ``signal.zscore`` is NaN, ``capital`` ≤ 0, or f* ≤ 0.

        :param Signal signal: Signal at the current bar.
        :param float price_y: Current price of the y leg.
        :param float price_x: Current price of the x leg.
        :param float hedge_ratio: Hedge ratio β_t (absolute value used).
        :param float capital: Current mark-to-market equity in monetary units.

        :returns: ``(qty_y, qty_x)`` — absolute quantities for each leg.
        :rtype: Tuple[float, float]
        """
        if signal.direction == Direction.FLAT or np.isnan(signal.zscore):
            return 0.0, 0.0
        if capital <= 0.0 or self._frac <= 0.0:
            return 0.0, 0.0
        return _quantities(self._frac, capital, price_y, price_x, hedge_ratio)


# ---------------------------------------------------------------------------
# Method 2 — KellyTruncatedExit
# ---------------------------------------------------------------------------

class KellyTruncatedExit(PositionSizer):
    """Kelly sizer where only the reversion level :math:`\\tilde{z}` is random.

    The exit z-score is modelled as a truncated standard normal:

    .. math::

        \\tilde{z} \\sim \\mathcal{N}(0,1) \\text{ truncated to }
        (-\\infty, z_{\\text{revert}}],
        \\quad p(\\tilde{z}) = \\frac{\\varphi(\\tilde{z})}{\\Phi(z_{\\text{revert}})}

    The observed entry :math:`z_t` is treated as a deterministic constant.
    The per-trade gain is :math:`G = z_t - \\tilde{z}`.

    Moments (:math:`\\lambda_- = \\lambda_-(z_{\\text{revert}})`):

    .. math::

        \\mathbb{E}[\\tilde{z}]  &= -\\lambda_- \\\\
        \\mathbb{E}[G]            &= z_t + \\lambda_- \\\\
        \\mathrm{Var}(\\tilde{z}) &= 1 - \\lambda_-(\\lambda_- + z_{\\text{revert}}) \\\\
        \\mathbb{E}[G^2]          &= \\mathrm{Var}(\\tilde{z}) + \\mathbb{E}[G]^2

    Second-order Kelly fraction (function of the observed :math:`z_t`):

    .. math::

        f^*(z_t) = \\frac{z_t + \\lambda_-}
                    {1 - \\lambda_-(\\lambda_- + z_{\\text{revert}}) + (z_t + \\lambda_-)^2}

    By symmetry of :math:`\\mathcal{N}(0,1)`, a LONG entry with
    :math:`z_t = -|z|` is equivalent to a SHORT entry with the reflected
    :math:`|z|`, so ``|signal.zscore|`` is used for both directions.
    The fraction is recomputed at each bar.

    :param float z_revert: Right-truncation point for the exit distribution
        (default 0.0, i.e. exit at the mean).
    :param float f_max: Hard cap on the Kelly fraction (default 0.5).
    """

    def __init__(
        self,
        z_revert: float = 0.0,
        f_max: float = 0.5,
    ) -> None:
        self.z_revert  = z_revert
        self.f_max     = f_max
        self._lam_minus = _mills_right(z_revert)
        self._var_ztilde = 1.0 - self._lam_minus * (self._lam_minus + z_revert)

    def _kelly(self, z_t: float) -> float:
        mu_g = z_t + self._lam_minus
        if mu_g <= 0.0:
            return 0.0
        e_g2 = self._var_ztilde + mu_g ** 2
        if e_g2 <= 0.0:
            return 0.0
        return float(min(mu_g / e_g2, self.f_max))

    def size(
        self,
        signal: Signal,
        price_y: float,
        price_x: float,
        hedge_ratio: float,
        capital: float = 0.0,
    ) -> Tuple[float, float]:
        """Compute quantities using the bar-dependent Kelly fraction f*(:math:`|z_t|`).

        Returns ``(0, 0)`` when ``signal.direction`` is ``FLAT``,
        ``signal.zscore`` is NaN, ``capital`` ≤ 0, or f*(:math:`|z_t|`) ≤ 0.

        :param Signal signal: Signal at the current bar.
        :param float price_y: Current price of the y leg.
        :param float price_x: Current price of the x leg.
        :param float hedge_ratio: Hedge ratio β_t (absolute value used).
        :param float capital: Current mark-to-market equity in monetary units.

        :returns: ``(qty_y, qty_x)`` — absolute quantities for each leg.
        :rtype: Tuple[float, float]
        """
        if signal.direction == Direction.FLAT or np.isnan(signal.zscore):
            return 0.0, 0.0
        if capital <= 0.0:
            return 0.0, 0.0
        frac = self._kelly(abs(signal.zscore))
        if frac <= 0.0:
            return 0.0, 0.0
        return _quantities(frac, capital, price_y, price_x, hedge_ratio)


# ---------------------------------------------------------------------------
# Method 3 — KellyTruncatedBoth
# ---------------------------------------------------------------------------

class KellyTruncatedBoth(PositionSizer):
    """Kelly sizer where both entry :math:`z` and exit :math:`\\tilde{z}` are
    random and independent.

    Entry and exit z-scores are modelled as independent truncated normals:

    .. math::

        z         &\\sim \\mathcal{N}(0,1) \\text{ truncated to }
                  [z_{\\text{entry}}, +\\infty) && \\text{(entry level)} \\\\
        \\tilde{z} &\\sim \\mathcal{N}(0,1) \\text{ truncated to }
                  (-\\infty, z_{\\text{revert}}] && \\text{(exit level)} \\\\
        z &\\perp \\tilde{z}

    The per-trade gain is :math:`G = z - \\tilde{z}`.

    Moments (:math:`\\lambda_+ = \\lambda_+(z_{\\text{entry}})`,
    :math:`\\lambda_- = \\lambda_-(z_{\\text{revert}})`):

    .. math::

        \\mathbb{E}[G]            &= \\lambda_+ + \\lambda_- \\\\
        \\mathrm{Var}(z)          &= 1 - \\lambda_+(\\lambda_+ - z_{\\text{entry}}) \\\\
        \\mathrm{Var}(\\tilde{z}) &= 1 - \\lambda_-(\\lambda_- + z_{\\text{revert}}) \\\\
        \\mathrm{Var}(G)          &= \\mathrm{Var}(z) + \\mathrm{Var}(\\tilde{z})
                                  && \\text{(independence)} \\\\
        \\mathbb{E}[G^2]          &= \\mathrm{Var}(G) + \\mathbb{E}[G]^2

    Second-order Kelly fraction:

    .. math::

        f^* = \\frac{\\lambda_+ + \\lambda_-}
               {2 - \\lambda_+(\\lambda_+ - z_{\\text{entry}})
               - \\lambda_-(\\lambda_- + z_{\\text{revert}})
               + (\\lambda_+ + \\lambda_-)^2}

    The fraction is constant and computed once at construction.

    :param float z_entry: Entry threshold; positions are opened when :math:`|z_t| \\geq z_{\\text{entry}}`.
    :param float z_revert: Right-truncation point for the exit distribution
        (default 0.0, i.e. exit at the mean).
    :param float f_max: Hard cap on the Kelly fraction (default 0.5).
    """

    def __init__(
        self,
        z_entry: float,
        z_revert: float = 0.0,
        f_max: float = 0.5,
    ) -> None:
        self.z_entry  = z_entry
        self.z_revert = z_revert
        self.f_max    = f_max
        self._frac    = self._kelly()

    def _kelly(self) -> float:
        lam_plus  = _mills_left(self.z_entry)
        lam_minus = _mills_right(self.z_revert)
        mu_g      = lam_plus + lam_minus
        var_z     = 1.0 - lam_plus  * (lam_plus  - self.z_entry)
        var_ztilde = 1.0 - lam_minus * (lam_minus + self.z_revert)
        e_g2      = var_z + var_ztilde + mu_g ** 2
        if e_g2 <= 0.0:
            return 0.0
        return float(min(mu_g / e_g2, self.f_max))

    def size(
        self,
        signal: Signal,
        price_y: float,
        price_x: float,
        hedge_ratio: float,
        capital: float = 0.0,
    ) -> Tuple[float, float]:
        """Compute quantities using the constant Kelly fraction f*.

        Returns ``(0, 0)`` when ``signal.direction`` is ``FLAT``,
        ``signal.zscore`` is NaN, ``capital`` ≤ 0, or f* ≤ 0.

        :param Signal signal: Signal at the current bar.
        :param float price_y: Current price of the y leg.
        :param float price_x: Current price of the x leg.
        :param float hedge_ratio: Hedge ratio β_t (absolute value used).
        :param float capital: Current mark-to-market equity in monetary units.

        :returns: ``(qty_y, qty_x)`` — absolute quantities for each leg.
        :rtype: Tuple[float, float]
        """
        if signal.direction == Direction.FLAT or np.isnan(signal.zscore):
            return 0.0, 0.0
        if capital <= 0.0 or self._frac <= 0.0:
            return 0.0, 0.0
        return _quantities(self._frac, capital, price_y, price_x, hedge_ratio)
