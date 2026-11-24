"""Daily returns: a synthetic regime-driven market, or real US stock prices.

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
from pathlib import Path

import numpy as np
import pandas as pd

REGIME_NAMES = ("bull", "neutral", "bear")


@dataclass(frozen=True)
class MarketConfig:
    n_assets: int = 12
    n_sectors: int = 3
    years: int = 15
    seed: int = 21
    transition: tuple = (
        (0.985, 0.010, 0.005),
        (0.015, 0.975, 0.010),
        (0.020, 0.030, 0.950),
    )
    market_mu: tuple = (0.0007, 0.0002, -0.0012)
    market_vol: tuple = (0.007, 0.010, 0.022)
    sector_vol: float = 0.004
    drift_persistence: float = 0.985
    drift_shock: float = 0.00008


def simulate_market(cfg: MarketConfig = MarketConfig()) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (daily returns DataFrame [dates x assets], regime label Series)."""
    rng = np.random.default_rng(cfg.seed)
    n, days = cfg.n_assets, 252 * cfg.years
    trans = np.asarray(cfg.transition)

    betas = rng.uniform(0.5, 1.5, n)
    idio_vol = rng.uniform(0.006, 0.014, n)
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


REAL_DATA = Path(__file__).resolve().parents[2] / "data" / "us_stocks_daily.csv"


def load_real_market(path: Path = REAL_DATA) -> pd.DataFrame:
    """Daily simple returns for 12 large US stocks from split- and dividend-adjusted closes.

    Covers April 2003 to April 2018 (about 15 years, 3,784 trading days). Prices come
    from the stock_prices.csv test file of PyPortfolioOpt (MIT licence). There are no
    regime labels on real data, so only returns are returned.
    """
    prices = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    return prices.pct_change().iloc[1:].dropna(how="any")
