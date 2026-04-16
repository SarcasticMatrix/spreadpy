from typing import Optional

import numpy as np
import pandas as pd
from dataclasses import dataclass

from spreadpy.data import PriceTimeSeries
from spreadpy.spread.hedgeRatioEstimator import HedgeRatioEstimator


@dataclass
class KalmanFilterWithVelocityParams:
    """
    Hyperparameters for the :class:`KalmanFilterWithVelocity` state-space model.

    Derived from an OLS bootstrap following Palomar (2024), §15.6.3, eq. (15.4).

    :param float sigma2_eps: Observation noise variance σ²_ε.
    :param float sigma2_mu: State noise variance for the intercept μ_t.
    :param float sigma2_gam: State noise variance for the hedge ratio γ_t.
    :param float sigma2_dgam: State noise variance for the velocity γ̇_t.
    :param float mu0: Initial intercept estimate μ₁ (from OLS).
    :param float gam0: Initial hedge ratio estimate γ₁ (from OLS).
    :param np.ndarray P0: Initial state covariance matrix (3×3, diagonal).
    """
    sigma2_eps:  float
    sigma2_mu:   float
    sigma2_gam:  float
    sigma2_dgam: float
    mu0:         float
    gam0:        float
    P0:          np.ndarray


class KalmanFilterWithVelocity(HedgeRatioEstimator):
    """
    Kalman filter hedge ratio estimator with augmented state [μ_t, γ_t, γ̇_t].

    Extends :class:`KalmanFilter` with a velocity state γ̇_t that allows
    the hedge ratio to follow a locally linear trend. Implements Palomar (2024),
    §15.6.3, eq. (15.4).

    Observation equation:

        y_t = μ_t + γ_t · x_t + ε_t,   ε_t ~ N(0, σ²_ε)

    Transition equations:

        μ_{t+1} = μ_t                  + η_{μ,t}
        γ_{t+1} = γ_t  + γ̇_t         + η_{γ,t}
        γ̇_{t+1} =       γ̇_t         + η_{γ̇,t}

    γ follows a locally linear trend driven by its velocity γ̇.
    The velocity is a slow random walk (σ²_γ̇ ≪ σ²_γ).

    The predictive hedge ratio γ_{t|t−1} is returned by :meth:`fit`.
    The normalised spread (no lookahead) is:

        z_t = (y_t − γ_{t|t−1} · x_t − μ_{t|t−1}) / (1 + γ_{t|t−1})

    Hyperparameters are initialised via OLS bootstrap (same heuristic as
    :class:`KalmanFilter`), with an additional term for σ²_γ̇:

        σ²_γ̇ = alpha_dgam · σ²_ε / Var[x]

    :param float alpha: State noise scale for μ and γ. Palomar recommends
        1e-6 for this model.
    :param Optional[float] alpha_dgam: State noise scale for γ̇. Should satisfy
        alpha_dgam < alpha so that velocity varies slowly.
        Defaults to alpha / 10.
    :param Optional[int] ls_window: Bars used for OLS initialisation.
        None uses the full series.
    """

    def __init__(
        self,
        alpha: float = 1e-6,
        alpha_dgam: Optional[float] = None,
        ls_window: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.alpha_dgam = alpha_dgam if alpha_dgam is not None else alpha / 10.0
        self.ls_window = ls_window

        # Exposés après fit()
        self.params_: Optional[KalmanFilterWithVelocityParams] = None
        self.mu_ts_:       Optional[pd.Series] = None
        self.velocity_ts_: Optional[pd.Series] = None
        self.normalized_spread_: Optional[pd.Series] = None

    # Matrice de transition F (3×3) — constante
    _F = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ])

    # ------------------------------------------------------------------
    # HedgeRatioEstimator interface
    # ------------------------------------------------------------------

    def fit(self, y: PriceTimeSeries, x: PriceTimeSeries) -> pd.Series:
        """
        Run the Kalman filter on (y, x) and return the one-step-ahead
        predictive hedge ratio γ_{t|t−1}.

        Using the predictive state rather than the filtered state γ_{t|t}
        ensures no lookahead bias: at bar t, γ_{t|t−1} is computed from
        observations {(y_s, x_s) : s ≤ t − 1} only.

        After calling this method the following attributes are set:

        - ``params_`` — fitted :class:`KalmanFilterWithVelocityParams`
        - ``mu_ts_`` — filtered intercept series μ_{t|t}
        - ``velocity_ts_`` — filtered velocity series γ̇_{t|t}
        - ``normalized_spread_`` — lookahead-free normalised spread

            z_t = (y_t − γ_{t|t−1} · x_t − μ_{t|t−1}) / (1 + γ_{t|t−1})

        :param PriceTimeSeries y: Dependent-leg price series.
        :param PriceTimeSeries x: Independent-leg price series.
        :returns: Predictive hedge ratio series γ_{t|t−1} aligned with ``y.index``.
        :rtype: pd.Series
        """
        y_al, x_al = y.align(x)
        yv = y_al.values.astype(float)
        xv = x_al.values.astype(float)

        params = self._init_params(yv, xv)
        self.params_ = params

        mu_filt, _, dgam_filt, gam_pred, mu_pred, _ = self._run_filter(
            yv, xv, params
        )

        index = y_al.index
        self.mu_ts_       = pd.Series(mu_filt,   index=index, name="mu")
        self.velocity_ts_ = pd.Series(dgam_filt, index=index, name="velocity")

        # Spread normalisé — exposé pour ZScoreSignal / CopulaSignal
        self.normalized_spread_ = pd.Series(
            (yv - gam_pred * xv - mu_pred) / (1.0 + gam_pred),
            index=index,
            name="normalized_spread",
        )

        return pd.Series(gam_pred, index=index, name="hedge_ratio")

    # ------------------------------------------------------------------
    # Initialisation LS
    # ------------------------------------------------------------------

    def _init_params(
        self, yv: np.ndarray, xv: np.ndarray
    ) -> KalmanFilterWithVelocityParams:
        """
        Bootstrap filter hyperparameters from OLS on the first T_ls bars.

        Extends the 2-state bootstrap (Palomar, §15.6.3) with an additional
        noise term for the velocity state γ̇_t:

            σ²_ε   = Var[ε^{OLS}]
            σ²_μ   = α      · σ²_ε
            σ²_γ   = α      · σ²_ε / Var[x]
            σ²_γ̇  = α_dgam · σ²_ε / Var[x]   (α_dgam < α ⟹ slow velocity)

        The initial state covariance P₀ is 3 × 3 diagonal:

            P₀ = diag(σ²_ε / T_ls,   σ²_ε / (T_ls · Var[x]),   σ²_γ)

        The third diagonal entry initialises uncertainty on the unknown
        velocity γ̇₁ at the level of the hedge-ratio process noise σ²_γ,
        reflecting that the initial trend is completely unknown.

        :param np.ndarray yv: Dependent-leg values, shape (T,).
        :param np.ndarray xv: Independent-leg values, shape (T,).
        :returns: Fitted :class:`KalmanFilterWithVelocityParams` dataclass.
        :rtype: KalmanFilterWithVelocityParams
        """
        T_ls = min(
            self.ls_window if self.ls_window is not None else len(yv),
            len(yv),
        )
        y_ls, x_ls = yv[:T_ls], xv[:T_ls]

        A = np.column_stack([np.ones(T_ls), x_ls])
        coef, *_ = np.linalg.lstsq(A, y_ls, rcond=None)
        mu_ls, gam_ls = float(coef[0]), float(coef[1])

        eps_ls     = y_ls - (mu_ls + gam_ls * x_ls)
        sigma2_eps = float(np.var(eps_ls, ddof=1))
        var_y2     = float(np.var(x_ls,   ddof=1))
        var_y2     = max(var_y2, 1e-12)

        sigma2_mu   = self.alpha      * sigma2_eps
        sigma2_gam  = self.alpha      * sigma2_eps / var_y2
        sigma2_dgam = self.alpha_dgam * sigma2_eps / var_y2

        # P0 3×3 diagonale : même logique que KalmanFilterWithDrift
        # + incertitude initiale sur γ̇ ≈ σ²_γ (on ne sait pas la trend initiale)
        P0 = np.diag([
            sigma2_eps / T_ls,
            sigma2_eps / (T_ls * var_y2),
            sigma2_gam,              # γ̇₁ inconnu → P₁[2,2] = σ²_γ
        ])

        return KalmanFilterWithVelocityParams(
            sigma2_eps=sigma2_eps,
            sigma2_mu=sigma2_mu,
            sigma2_gam=sigma2_gam,
            sigma2_dgam=sigma2_dgam,
            mu0=mu_ls,
            gam0=gam_ls,
            P0=P0,
        )

    # ------------------------------------------------------------------
    # Boucle de filtrage
    # ------------------------------------------------------------------

    def _run_filter(
        self,
        yv: np.ndarray,
        xv: np.ndarray,
        p:  KalmanFilterWithVelocityParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the forward Kalman recursion for the 3-state model.

        State vector: α_t = [μ_t, γ_t, γ̇_t]^T.

        Transition matrix (locally-linear trend on γ):

            F = [[1, 0, 0],
                 [0, 1, 1],
                 [0, 0, 1]]

        State noise: Q = diag(σ²_μ, σ²_γ, σ²_γ̇).
        Observation vector at bar t: H_t = [1, x_t, 0].

        For t = 1, …, T:

        **Predict**::

            α_{t|t−1} = F · α_{t−1|t−1}
            P_{t|t−1} = F · P_{t−1|t−1} · F^T + Q

        which gives γ_{t|t−1} = γ_{t−1|t−1} + γ̇_{t−1|t−1} (trend extrapolation).

        **Update**::

            ν_t  = y_t − H_t · α_{t|t−1}                 (innovation, scalar)
            S_t  = H_t · P_{t|t−1} · H_t^T + σ²_ε        (innovation variance)
            K_t  = P_{t|t−1} · H_t^T / S_t                (Kalman gain, 3×1)
            α_{t|t} = α_{t|t−1} + K_t · ν_t
            P_{t|t} = (I − K_t · H_t^T) · P_{t|t−1}

        :param np.ndarray yv: Dependent-leg values, shape (T,).
        :param np.ndarray xv: Independent-leg values, shape (T,).
        :param KalmanFilterWithVelocityParams p: Hyperparameters.
        :returns: Tuple
            ``(μ_{t|t}, γ_{t|t}, γ̇_{t|t}, γ_{t|t−1}, μ_{t|t−1}, S_t)``,
            each an ndarray of shape (T,).
        :rtype: tuple[np.ndarray, ...]
        """
        n   = len(yv)
        F   = self._F
        Q   = np.diag([p.sigma2_mu, p.sigma2_gam, p.sigma2_dgam])
        R   = p.sigma2_eps

        mu_filt   = np.empty(n)
        gam_filt  = np.empty(n)
        dgam_filt = np.empty(n)
        mu_pred   = np.empty(n)
        gam_pred  = np.empty(n)
        S_inn = np.empty(n)

        # État initial :  s = [μ₁, γ₁, γ̇₁=0]
        s = np.array([p.mu0, p.gam0, 0.0])
        P = p.P0.copy()

        for t in range(n):
            # ── Predict ─────────────────────────────────────────────
            s_pred = F @ s
            P_pred = F @ P @ F.T + Q

            mu_pred[t]  = s_pred[0]
            gam_pred[t] = s_pred[1]
            # (γ dérivé prédictif non exposé mais disponible via s_pred[2])

            # ── Update ──────────────────────────────────────────────
            # H_t = [1, y2_t, 0]  →  observation = μ_t + γ_t·y2_t
            H = np.array([1.0, xv[t], 0.0])

            innovation = yv[t] - H @ s_pred
            S_inn[t]      = H @ P_pred @ H + R
            K          = (P_pred @ H) / S_inn[t]   # (3,)

            s = s_pred + K * innovation
            P = (np.eye(3) - np.outer(K, H)) @ P_pred

            mu_filt[t]   = s[0]
            gam_filt[t]  = s[1]
            dgam_filt[t] = s[2]

        return mu_filt, gam_filt, dgam_filt, gam_pred, mu_pred, S_inn