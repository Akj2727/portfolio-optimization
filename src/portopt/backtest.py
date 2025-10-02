"""Monthly-rebalanced backtest with drifting weights and turnover costs."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .risk import cvar

TRADING_DAYS = 252


def run_backtest(returns: pd.DataFrame, rebalance_dates: pd.DatetimeIndex,
                 weight_fn: Callable[[pd.Timestamp], np.ndarray],
                 cost_bps: float = 10.0) -> pd.Series:
    """Daily portfolio returns. Weights are set at the close of each rebalance date
    and earn returns from the next day; between rebalances they drift with prices."""
    start = rebalance_dates[0]
    days = returns.index[returns.index > start]
    reb = set(rebalance_dates)
    w = np.zeros(returns.shape[1])
    held = w.copy()
    out = []
    pending_cost = 0.0
    w = weight_fn(start)
    pending_cost = np.abs(w - held).sum() * cost_bps / 1e4
    held = w.copy()
    for d in days:
        r = returns.loc[d].values
        port = float(held @ r) - pending_cost
        pending_cost = 0.0
        held = held * (1 + r)
        held = held / held.sum()
        out.append(port)
        if d in reb:
            target = weight_fn(d)
            pending_cost = np.abs(target - held).sum() * cost_bps / 1e4
            held = target.copy()
    return pd.Series(out, index=days)


def summarize(daily: pd.Series) -> dict:
    ann_ret = (1 + daily).prod() ** (TRADING_DAYS / len(daily)) - 1
    ann_vol = daily.std() * np.sqrt(TRADING_DAYS)
    equity = (1 + daily).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": daily.mean() / daily.std() * np.sqrt(TRADING_DAYS),
        "max_drawdown": max_dd,
        "cvar_95_daily": cvar(daily.values),
    }
