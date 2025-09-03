"""Synthetic daily returns for a universe of assets driven by hidden market regimes.

The generator has three ingredients a real market also has, so the methods downstream
have something to find:
  * a 3-state Markov chain of market regimes (bull / neutral / bear) with different
    market drift and volatility,
  * a factor structure: market beta, sector factors and idiosyncratic noise,
  * a slow, persistent per-asset drift (AR(1)), which makes trailing returns weakly
    predictive of future returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIME_NAMES = ("bull", "neutral", "bear")


@dataclass(frozen=True)
class MarketConfig:
    n_assets: int = 12
    n_sectors: int = 3
    years: int = 15
    seed: int = 7
    transition: tuple = (
        (0.985, 0.010, 0.005),
        (0.015, 0.975, 0.010),
        (0.020, 0.030, 0.950),
    )
    market_mu: tuple = (0.0007, 0.0002, -0.0012)
    market_vol: tuple = (0.007, 0.010, 0.022)
    sector_vol: float = 0.004
    drift_persistence: float = 0.985
    drift_shock: float = 0.00012


def simulate_market(cfg: MarketConfig = MarketConfig()) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (daily returns DataFrame [dates x assets], regime label Series)."""
    rng = np.random.default_rng(cfg.seed)
    n, days = cfg.n_assets, 252 * cfg.years
    trans = np.asarray(cfg.transition)

    betas = rng.uniform(0.5, 1.5, n)
    idio_vol = rng.uniform(0.007, 0.018, n)
    sector = np.arange(n) % cfg.n_sectors

    regime = np.empty(days, dtype=int)
    regime[0] = 1
    for t in range(1, days):
        regime[t] = rng.choice(3, p=trans[regime[t - 1]])

    mkt = rng.normal(np.take(cfg.market_mu, regime), np.take(cfg.market_vol, regime))
    sec = rng.normal(0.0, cfg.sector_vol, (days, cfg.n_sectors))[:, sector]

    drift = np.zeros((days, n))
    for t in range(1, days):
        drift[t] = cfg.drift_persistence * drift[t - 1] + rng.normal(0.0, cfg.drift_shock, n)

    rets = mkt[:, None] * betas + sec + drift + rng.normal(0.0, idio_vol, (days, n))

    dates = pd.bdate_range("2011-01-03", periods=days)
    cols = [f"A{i:02d}" for i in range(n)]
    returns = pd.DataFrame(rets, index=dates, columns=cols)
    regimes = pd.Series(np.take(REGIME_NAMES, regime), index=dates, name="regime")
    return returns, regimes
