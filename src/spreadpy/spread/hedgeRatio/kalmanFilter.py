from typing import Optional

import numpy as np
import pandas as pd
from dataclasses import dataclass

from spreadpy.data import PriceTimeSeries
from spreadpy.spread.hedgeRatioEstimator import HedgeRatioEstimator


@dataclass
class KalmanFilterParams:
    """
    Hyperparameters for the :class:`KalmanFilter` state-space model.

    Derived from an OLS bootstrap on the first ``ls_window`` bars,
    following the heuristic of Palomar (2024), §15.6.3.

    :param float sigma2_eps: Observation noise variance σ²_ε.
    :param float sigma2_mu: State noise variance for the intercept,
        σ²_μ = α · σ²_ε.
    :param float sigma2_gam: State noise variance for the hedge ratio,
        σ²_γ = α · σ²_ε / Var[x].
    :param float mu0: Initial intercept estimate μ₁ (from OLS).
    :param float gam0: Initial hedge ratio estimate γ₁ (from OLS).
    :param np.ndarray P0: Initial state covariance matrix (2×2, diagonal).
    """
    sigma2_eps: float
    sigma2_mu:  float
    sigma2_gam: float
    mu0:        float
    gam0:       float
    P0:         np.ndarray


class KalmanFilter(HedgeRatioEstimator):
    """
    Kalman filter hedge ratio estimator with two latent states [μ_t, γ_t].

    Implements the pairs-trading state-space model of Palomar (2024), §15.6.3, eq. (15.3).

    Observation equation:

        y_t = μ_t + γ_t · x_t + ε_t,   ε_t ~ N(0, σ²_ε)

    Transition equations (independent random walks):

        μ_{t+1} = μ_t + η_{μ,t},   η_{μ,t} ~ N(0, σ²_μ)
        γ_{t+1} = γ_t + η_{γ,t},   η_{γ,t} ~ N(0, σ²_γ)

    The hedge ratio returned by :meth:`fit` is the one-step-ahead predictive
    γ_{t|t−1}, free of lookahead bias. The normalised spread is:

        z_t = (y_t − γ_{t|t−1} · x_t − μ_{t|t−1}) / (1 + γ_{t|t−1})

    Hyperparameters are initialised via an OLS bootstrap on the first
    ``ls_window`` bars (Palomar, §15.6.3):

        σ²_ε = Var[ε^OLS],   σ²_μ = α · σ²_ε,   σ²_γ = α · σ²_ε / Var[x]

    :param float alpha: Controls state noise relative to observation noise.
        Smaller values → slower adaptation of μ and γ. Typical range: 1e-6 to 1e-4.
        Palomar uses 1e-5 for this model.
    :param Optional[int] ls_window: Number of bars for the OLS initialisation.
        None uses the entire series (recommended: pass ``train_size`` from the engine).
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