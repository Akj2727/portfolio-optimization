"""Historical-simulation CVaR (expected shortfall)."""
from __future__ import annotations

import numpy as np

ALPHA = 0.95


def _tail_size(n: int, alpha: float) -> int:
    # round first: (1 - 0.95) * 1000 is 50.000000000000004 in floating point
    return max(1, int(np.ceil(round((1 - alpha) * n, 9))))


def cvar(portfolio_returns: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Mean loss in the worst (1 - alpha) tail, as a positive number.

    Accepts a 1-D array of returns, or a 2-D array [scenarios x portfolios] and
    returns one CVaR per column.
    """
    r = np.asarray(portfolio_returns, dtype=float)
    one_d = r.ndim == 1
    if one_d:
        r = r[:, None]
    k = _tail_size(r.shape[0], alpha)
    worst = np.partition(r, k - 1, axis=0)[:k]
    out = -worst.mean(axis=0)
    return out[0] if one_d else out


def var(portfolio_returns: np.ndarray, alpha: float = ALPHA) -> float:
    r = np.sort(np.asarray(portfolio_returns, dtype=float))
    k = _tail_size(len(r), alpha)
    return float(-r[k - 1])
