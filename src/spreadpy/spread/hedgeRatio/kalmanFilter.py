from typing import Optional

import numpy as np
import pandas as pd
from dataclasses import dataclass

from spreadpy.data import PriceTimeSeries
from spreadpy.spread.hedgeRatioEstimator import HedgeRatioEstimator


@dataclass
class KalmanFilterParams:
    """
    Paramètres du modèle state-space (Palomar §15.6.3, eq. 15.3).

    sigma2_eps : variance du bruit d'observation  ε_t ~ N(0, σ²_ε)
    sigma2_mu  : variance du bruit d'état sur μ_t ~ N(0, σ²_μ)
    sigma2_gam : variance du bruit d'état sur γ_t ~ N(0, σ²_γ)
    mu0, gam0  : états initiaux
    P0         : matrice de covariance initiale (2×2)
    """
    sigma2_eps: float
    sigma2_mu:  float
    sigma2_gam: float
    mu0:        float
    gam0:       float
    P0:         np.ndarray


class KalmanFilter(HedgeRatioEstimator):
    """
    Kalman filter pairs trading — Palomar (2024), §15.6.3, eq. (15.3).

    State-space model
    -----------------
    Observation :
        y1_t = [1, y2_t] @ [μ_t, γ_t]ᵀ + ε_t,   ε_t ~ N(0, σ²_ε)

    Transition (random walk sur les deux états) :
        [μ_{t+1}]   [1 0] [μ_t]   [η_1t]
        [γ_{t+1}] = [0 1] [γ_t] + [η_2t]

        η_1t ~ N(0, σ²_μ),  η_2t ~ N(0, σ²_γ)

    Le spread normalisé (leverage 1) vaut :
        z_t = (y1_t - γ_{t|t-1}·y2_t - μ_{t|t-1}) / (1 + γ_{t|t-1})

    Initialisation des hyperparamètres (heuristique LS de Palomar)
    --------------------------------------------------------------
    On estime (μ_LS, γ_LS) par OLS sur les `ls_window` premières barres, puis :
        σ²_ε      = Var[ε^LS]
        σ²_μ      = α · Var[ε^LS]
        σ²_γ      = α · Var[ε^LS] / Var[y2]
        μ₁  ~ N(μ_LS,  σ²_ε / T_LS)
        γ₁  ~ N(γ_LS,  σ²_ε / (T_LS · Var[y2]))

    Parameters
    ----------
    alpha : float
        Hyper-paramètre α — ratio variabilité des états / variabilité du spread.
        Palomar utilise α=1e-5 (basic) et α=1e-6 (avec momentum).
        Valeurs typiques : 1e-6 à 1e-4.
    ls_window : int
        Nombre de barres utilisées pour l'initialisation LS.
        Si None, utilise la série entière (cohérent avec le walk-forward :
        passer la fenêtre de train depuis l'engine).
    """

    def __init__(
        self,
        alpha: float = 1e-5,
        ls_window: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.ls_window = ls_window

        # Exposés après fit()
        self.params_: Optional[KalmanFilterParams] = None
        self.mu_ts_:  Optional[pd.Series] = None   # série μ_{t|t}
        self.normalized_spread_: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # HedgeRatioEstimator interface
    # ------------------------------------------------------------------

    def fit(self, y: PriceTimeSeries, x: PriceTimeSeries) -> pd.Series:
        """
        Ajuste le filtre et retourne γ_{t|t-1} (hedge ratio prédictif,
        sans lookahead — utilisé pour construire le spread dans SpreadSeries).
        """
        y_al, x_al = y.align(x)
        yv = y_al.values.astype(float)   # y1
        xv = x_al.values.astype(float)   # y2

        params = self._init_params(yv, xv)
        self.params_ = params

        mu_filt, gam_filt, gam_pred, mu_pred = self._run_filter(yv, xv, params)

        index = y_al.index
        self.mu_ts_ = pd.Series(mu_filt, index=index, name="mu")

        # Spread normalisé (Palomar eq. après 15.3) — exposé pour le signal
        self.normalized_spread_ = pd.Series(
            (yv - gam_pred * xv - mu_pred) / (1.0 + gam_pred),
            index=index,
            name="normalized_spread",
        )

        # On retourne γ_{t|t-1} : prédictif, sans lookahead
        return pd.Series(gam_pred, index=index, name="hedge_ratio")

    # ------------------------------------------------------------------
    # Initialisation LS (heuristique Palomar §15.6.3)
    # ------------------------------------------------------------------

    def _init_params(self, yv: np.ndarray, xv: np.ndarray) -> KalmanFilterParams:
        T_ls = self.ls_window if self.ls_window is not None else len(yv)
        T_ls = min(T_ls, len(yv))

        y_ls = yv[:T_ls]
        x_ls = xv[:T_ls]

        # OLS : y1 ≈ μ + γ·y2
        A = np.column_stack([np.ones(T_ls), x_ls])
        coef, *_ = np.linalg.lstsq(A, y_ls, rcond=None)
        mu_ls, gam_ls = float(coef[0]), float(coef[1])

        eps_ls    = y_ls - (mu_ls + gam_ls * x_ls)
        sigma2_eps = float(np.var(eps_ls, ddof=1))
        var_y2     = float(np.var(x_ls,   ddof=1))

        # Bruits d'état
        sigma2_mu  = self.alpha * sigma2_eps
        sigma2_gam = self.alpha * sigma2_eps / max(var_y2, 1e-12)

        # Covariance initiale P0 (diagonale, formule Palomar)
        P0 = np.diag([
            sigma2_eps / T_ls,
            sigma2_eps / (T_ls * max(var_y2, 1e-12)),
        ])

        return KalmanFilterParams(
            sigma2_eps=sigma2_eps,
            sigma2_mu=sigma2_mu,
            sigma2_gam=sigma2_gam,
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
        p:  KalmanFilterParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retourne (mu_filtered, gam_filtered, gam_predicted, mu_predicted).

        mu/gam_filtered  : α_{t|t}   — mis à jour avec l'observation t
        mu/gam_predicted : α_{t|t-1} — avant mise à jour  (no lookahead)
        """
        n = len(yv)

        # Matrices fixes du modèle
        F = np.eye(2)                           # transition (random walk)
        Q = np.diag([p.sigma2_mu, p.sigma2_gam])  # bruit d'état
        R = p.sigma2_eps                        # bruit d'observation (scalaire)

        # Stockage
        mu_filt  = np.empty(n)
        gam_filt = np.empty(n)
        mu_pred  = np.empty(n)
        gam_pred = np.empty(n)

        # État initial
        s = np.array([p.mu0, p.gam0])   # [μ₁, γ₁]
        P = p.P0.copy()

        for t in range(n):
            # ── Predict ─────────────────────────────────────────────
            s_pred = F @ s          # = s  (random walk ⟹ F = I)
            P_pred = F @ P @ F.T + Q

            # Sauvegarde des prédictifs (no lookahead)
            mu_pred[t]  = s_pred[0]
            gam_pred[t] = s_pred[1]

            # ── Update ──────────────────────────────────────────────
            # Vecteur d'observation H_t = [1, y2_t]
            H = np.array([1.0, xv[t]])

            innovation = yv[t] - H @ s_pred          # scalaire
            S_inn      = H @ P_pred @ H + R           # variance innovation
            K          = (P_pred @ H) / S_inn         # gain de Kalman (2×1)

            s = s_pred + K * innovation
            P = (np.eye(2) - np.outer(K, H)) @ P_pred

            mu_filt[t]  = s[0]
            gam_filt[t] = s[1]

        return mu_filt, gam_filt, gam_pred, mu_pred