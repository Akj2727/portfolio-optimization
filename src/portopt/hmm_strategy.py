"""Regime-switching benchmarks built on a Gaussian HMM of the market.

The HMM is fit once on the training period's equal-weight market returns. At each
rebalance we compute the filtered regime probabilities from data up to that day only
(the last row of forward-backward on a past window equals the forward filter).

Two strategies:
  * HMM-Blend:  mix a risk-on, a neutral and a defensive portfolio by regime probability.
  * HMM-Switch: hold the risk-on portfolio unless P(bear) > 0.5, then go defensive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class MarketHMM:
    def __init__(self, n_states: int = 3, seed: int = 0, window: int = 504):
        self.model = GaussianHMM(n_components=n_states, covariance_type="full",
                                 n_iter=200, random_state=seed)
        self.window = window
        self.order = None  # state indices sorted bull -> bear

    @staticmethod
    def _obs(mkt: pd.Series) -> np.ndarray:
        # daily market return and its 5-day absolute average as a volatility proxy
        return np.column_stack([mkt.values, mkt.abs().rolling(5, min_periods=1).mean().values])

    def fit(self, mkt_train: pd.Series) -> "MarketHMM":
        self.model.fit(self._obs(mkt_train))
        self.order = np.argsort(-self.model.means_[:, 0])
        return self

    def filtered_probs(self, mkt_history: pd.Series) -> np.ndarray:
        """P(bull, neutral, bear) on the last day of mkt_history."""
        obs = self._obs(mkt_history.iloc[-self.window:])
        return self.model.predict_proba(obs)[-1][self.order]


def inverse_vol(returns: pd.DataFrame) -> np.ndarray:
    w = 1.0 / returns.std().values
    return w / w.sum()


def beta_tilt(returns: pd.DataFrame) -> np.ndarray:
    mkt = returns.mean(axis=1)
    beta = returns.apply(lambda c: c.cov(mkt)).values / mkt.var()
    w = np.clip(beta, 0.05, None)
    return w / w.sum()


def regime_portfolios(lookback: pd.DataFrame) -> dict[str, np.ndarray]:
    n = lookback.shape[1]
    return {
        "risk_on": beta_tilt(lookback),
        "neutral": np.full(n, 1.0 / n),
        "defensive": inverse_vol(lookback),
    }


def blend_weights(probs: np.ndarray, ports: dict[str, np.ndarray]) -> np.ndarray:
    return probs[0] * ports["risk_on"] + probs[1] * ports["neutral"] + probs[2] * ports["defensive"]


def switch_weights(probs: np.ndarray, ports: dict[str, np.ndarray]) -> np.ndarray:
    return ports["defensive"] if probs[2] > 0.5 else ports["risk_on"]
