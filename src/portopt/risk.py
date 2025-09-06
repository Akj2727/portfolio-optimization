"""Historical-simulation CVaR (expected shortfall)."""
from __future__ import annotations

import numpy as np

ALPHA = 0.95


def cvar(portfolio_returns: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Mean loss in the worst (1 - alpha) tail, as a positive number.

    Accepts a 1-D array of returns, or a 2-D array [scenarios x portfolios] and
    returns one CVaR per column.
    """
    r = np.asarray(portfolio_returns, dtype=float)
    one_d = r.ndim == 1
    if one_d:
        r = r[:, None]
    k = max(1, int(np.ceil((1 - alpha) * r.shape[0])))
    worst = np.partition(r, k - 1, axis=0)[:k]
    out = -worst.mean(axis=0)
    return out[0] if one_d else out


def var(portfolio_returns: np.ndarray, alpha: float = ALPHA) -> float:
    r = np.sort(np.asarray(portfolio_returns, dtype=float))
    k = max(1, int(np.ceil((1 - alpha) * len(r))))
    return float(-r[k - 1])
