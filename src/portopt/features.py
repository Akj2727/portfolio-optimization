"""Point-in-time features and forward-return targets.

Every feature on date t uses returns up to and including t. The target on date t is
the sum of returns from t+1 to t+horizon, so it never overlaps the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 21  # one trading month


def make_features(returns: pd.DataFrame) -> pd.DataFrame:
    mkt = returns.mean(axis=1)
    frames = {
        "mom_5": returns.rolling(5).sum(),
        "mom_21": returns.rolling(21).sum(),
        "mom_63": returns.rolling(63).sum(),
        "mom_126": returns.rolling(126).sum(),
        "vol_21": returns.rolling(21).std(),
        "vol_63": returns.rolling(63).std(),
        "rel_mom_63": returns.rolling(63).sum().sub(mkt.rolling(63).sum(), axis=0),
        "beta_126": returns.rolling(126).cov(mkt).div(mkt.rolling(126).var(), axis=0),
    }
    per_asset = pd.concat({k: v.stack() for k, v in frames.items()}, axis=1)
    per_asset.index.names = ["date", "asset"]
    market = pd.DataFrame(
        {"mkt_mom_21": mkt.rolling(21).sum(), "mkt_vol_21": mkt.rolling(21).std()}
    )
    out = per_asset.join(market, on="date")
    return out.dropna()


def make_target(returns: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    fwd = returns[::-1].rolling(horizon).sum()[::-1].shift(-1)
    s = fwd.stack()
    s.index.names = ["date", "asset"]
    return s.rename("fwd_ret").dropna()


def aligned(returns: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.Series]:
    X = make_features(returns)
    y = make_target(returns, horizon)
    idx = X.index.intersection(y.index)
    return X.loc[idx], y.loc[idx]


def cross_sectional_ic(pred: pd.Series, actual: pd.Series) -> float:
    """Mean Spearman rank correlation between prediction and outcome across dates."""
    df = pd.concat({"p": pred, "a": actual}, axis=1).dropna()
    ics = df.groupby(level="date").apply(lambda g: g["p"].rank().corr(g["a"].rank()))
    return float(np.nanmean(ics))
