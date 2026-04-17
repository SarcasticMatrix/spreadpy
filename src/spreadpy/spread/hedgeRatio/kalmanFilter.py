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

    .. note::
        **Use log-prices.** The filter assumes constant observation noise σ²_ε.
        With raw prices, σ²_ε ∝ price², making the Kalman gain K_t systematically
        mis-calibrated when prices drift. Passing ``log(y)``, ``log(x)`` makes
        σ²_ε approximately homoscedastic and gives more stable hedge ratio estimates.
        In :class:`BacktestEngine`, set ``log_prices=True`` to apply this
        automatically while keeping actual prices for P&L accounting.
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
        self.mu_ts_:             Optional[pd.Series] = None
        self.normalized_spread_: Optional[pd.Series] = None
        self.innovations_ts_:    Optional[pd.Series] = None   # ν_t
        self.innovation_var_ts_: Optional[pd.Series] = None   # S_t

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

        - ``params_`` — fitted :class:`KalmanFilterParams`
        - ``mu_ts_`` — filtered intercept series μ_{t|t}
        - ``normalized_spread_`` — lookahead-free normalised spread

            z_t = (y_t − γ_{t|t−1} · x_t − μ_{t|t−1}) / (1 + γ_{t|t−1})

        :param PriceTimeSeries y: Dependent-leg price series.
        :param PriceTimeSeries x: Independent-leg price series.
        :returns: Predictive hedge ratio series γ_{t|t−1} aligned with ``y.index``.
        :rtype: pd.Series
        """
        y_al, x_al = y.align(x)
        yv = y_al.values.astype(float)   # y1
        xv = x_al.values.astype(float)   # y2

        params = self._init_params(yv, xv)
        self.params_ = params

        mu_filt, gam_filt, gam_pred, mu_pred, innovations, S_inn = self._run_filter(yv, xv, params)

        index = y_al.index
        self.mu_ts_             = pd.Series(mu_filt,     index=index, name="mu")
        self.innovations_ts_    = pd.Series(innovations, index=index, name="innovation")
        self.innovation_var_ts_ = pd.Series(S_inn,       index=index, name="innovation_var")

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
        """
        Bootstrap filter hyperparameters from OLS on the first T_ls bars.

        Following Palomar (2024), §15.6.3, the noise variances are set as:

            σ²_ε = Var[ε^{OLS}]   (sample variance of OLS residuals, ddof=1)
            σ²_μ = α · σ²_ε
            σ²_γ = α · σ²_ε / Var[x]

        The initial state covariance P₀ is diagonal with entries derived
        from the precision of the OLS intercept and slope estimates:

            P₀ = diag(σ²_ε / T_ls,   σ²_ε / (T_ls · Var[x]))

        :param np.ndarray yv: Dependent-leg values, shape (T,).
        :param np.ndarray xv: Independent-leg values, shape (T,).
        :returns: Fitted :class:`KalmanFilterParams` dataclass.
        :rtype: KalmanFilterParams
        """
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
        Execute the forward Kalman recursion for the 2-state model.

        State vector: α_t = [μ_t, γ_t]^T.
        Transition matrix: F = I₂ (independent random walks).
        State noise: Q = diag(σ²_μ, σ²_γ).
        Observation vector at bar t: H_t = [1, x_t].

        For t = 1, …, T:

        **Predict**::

            α_{t|t−1} = F · α_{t−1|t−1}
            P_{t|t−1} = F · P_{t−1|t−1} · F^T + Q

        **Update**::

            ν_t  = y_t − H_t · α_{t|t−1}                 (innovation)
            S_t  = H_t · P_{t|t−1} · H_t^T + σ²_ε        (innovation variance)
            K_t  = P_{t|t−1} · H_t^T / S_t                (Kalman gain, 2×1)
            α_{t|t} = α_{t|t−1} + K_t · ν_t
            P_{t|t} = (I − K_t · H_t^T) · P_{t|t−1}      (Joseph form omitted)

        :param np.ndarray yv: Dependent-leg values, shape (T,).
        :param np.ndarray xv: Independent-leg values, shape (T,).
        :param KalmanFilterParams p: Hyperparameters from :meth:`_init_params`.
        :returns: Tuple ``(μ_{t|t}, γ_{t|t}, γ_{t|t−1}, μ_{t|t−1})``,
            each an ndarray of shape (T,).
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        n = len(yv)

        # Matrices fixes du modèle
        F = np.eye(2)                           # transition (random walk)
        Q = np.diag([p.sigma2_mu, p.sigma2_gam])  # bruit d'état
        R = p.sigma2_eps                        # bruit d'observation (scalaire)

        # Stockage
        mu_filt     = np.empty(n)
        gam_filt    = np.empty(n)
        mu_pred     = np.empty(n)
        gam_pred    = np.empty(n)
        innovations = np.empty(n)
        S_inn       = np.empty(n)

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

            innovations[t] = yv[t] - H @ s_pred
            S_inn[t]       = H @ P_pred @ H + R
            K              = (P_pred @ H) / S_inn[t]

            s = s_pred + K * innovations[t]
            P = (np.eye(2) - np.outer(K, H)) @ P_pred

            mu_filt[t]  = s[0]
            gam_filt[t] = s[1]

        return mu_filt, gam_filt, gam_pred, mu_pred, innovations, S_inn