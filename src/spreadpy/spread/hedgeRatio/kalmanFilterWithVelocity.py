from typing import Optional

import numpy as np
import pandas as pd
from dataclasses import dataclass

from spreadpy.data import PriceTimeSeries
from spreadpy.spread.hedgeRatioEstimator import HedgeRatioEstimator


@dataclass
class KalmanFilterWithVelocityParams:
    """
    Paramètres du modèle state-space (Palomar §15.6.3, eq. 15.4).

    sigma2_eps  : variance du bruit d'observation
    sigma2_mu   : variance du bruit d'état sur μ_t
    sigma2_gam  : variance du bruit d'état sur γ_t
    sigma2_dgam : variance du bruit d'état sur γ̇_t (vélocité)
    mu0, gam0   : états initiaux (issus du LS bootstrap)
    P0          : matrice de covariance initiale (3×3)
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
    Kalman filter avec état augmenté [μ_t, γ_t, γ̇_t] — Palomar (2024),
    §15.6.3, eq. (15.4).

    State-space model
    -----------------
    Observation :
        y1_t = [1, y2_t, 0] @ [μ_t, γ_t, γ̇_t]ᵀ + ε_t,  ε_t ~ N(0, σ²_ε)

    Transition :
        [μ_{t+1}  ]   [1  0  0] [μ_t  ]   [η_μ,t  ]
        [γ_{t+1}  ] = [0  1  1] [γ_t  ] + [η_γ,t  ]
        [γ̇_{t+1} ]   [0  0  1] [γ̇_t ]   [η_γ̇,t ]

        γ_{t+1} = γ_t + γ̇_t + η_γ,t   →  le hedge ratio suit son trend
        γ̇_{t+1} = γ̇_t       + η_γ̇,t  →  la vélocité évolue lentement

    Spread normalisé (Palomar, après eq. 15.4) :
        z_t = (y1_t - γ_{t|t-1}·y2_t - μ_{t|t-1}) / (1 + γ_{t|t-1})

    Initialisation (heuristique LS identique à KalmanFilterWithDrift)
    -----------------------------------------------------------------
    OLS sur `ls_window` barres → (μ_LS, γ_LS, ε^LS), puis :
        σ²_ε       = Var[ε^LS]
        σ²_μ       = α      · Var[ε^LS]
        σ²_γ       = α      · Var[ε^LS] / Var[y2]
        σ²_γ̇      = α_dgam · Var[ε^LS] / Var[y2]
        γ̇₁ ~ N(0, σ²_γ)   (vélocité initiale nulle, incertitude ~ celle de γ)

    Parameters
    ----------
    alpha : float
        Hyper-paramètre α pour σ²_μ et σ²_γ.
        Palomar recommande α=1e-6 pour le modèle avec momentum.
    alpha_dgam : float
        Hyper-paramètre pour le bruit de la vélocité γ̇.
        En général alpha_dgam < alpha pour que γ̇ varie lentement.
        Défaut : alpha / 10.
    ls_window : int | None
        Nombre de barres pour l'initialisation LS.
        None → série entière (passer train_size depuis le WalkForwardEngine).
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
        Ajuste le filtre et retourne γ_{t|t-1} (prédictif, sans lookahead).
        """
        y_al, x_al = y.align(x)
        yv = y_al.values.astype(float)
        xv = x_al.values.astype(float)

        params = self._init_params(yv, xv)
        self.params_ = params

        mu_filt, gam_filt, dgam_filt, gam_pred, mu_pred, S_inn = self._run_filter(
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retourne (mu_filt, gam_filt, dgam_filt, gam_pred, mu_pred).
        Les *_pred sont les prédictifs α_{t|t-1} (no lookahead).
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