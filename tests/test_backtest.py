import numpy as np
import pandas as pd

from portopt.backtest import run_backtest, summarize


def _returns(n_days=60, n_assets=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(rng.normal(0.0005, 0.01, (n_days, n_assets)), index=idx)


def test_single_asset_buy_and_hold_matches_asset():
    r = _returns()
    reb = r.index[::20]
    w = np.array([1.0, 0.0, 0.0])
    daily = run_backtest(r, reb, lambda d: w, cost_bps=0.0)
    assert np.allclose(daily.values, r.loc[daily.index, 0].values)


def test_weights_never_see_same_day_return():
    r = _returns()
    reb = r.index[::20]
    seen = []
    run_backtest(r, reb, lambda d: seen.append(d) or np.ones(3) / 3, cost_bps=0.0)
    # each weight decision is taken on a rebalance date, and returns start the day after
    assert seen[0] == reb[0]


def test_costs_reduce_return():
    r = _returns()
    reb = r.index[::5]
    flip = iter([np.array([1, 0, 0.0]), np.array([0, 1, 0.0])] * 50)
    a = run_backtest(r, reb, lambda d: next(flip), cost_bps=0.0)
    flip = iter([np.array([1, 0, 0.0]), np.array([0, 1, 0.0])] * 50)
    b = run_backtest(r, reb, lambda d: next(flip), cost_bps=20.0)
    assert b.sum() < a.sum()


def test_summary_fields():
    s = summarize(pd.Series(np.random.default_rng(1).normal(0.0004, 0.01, 1000)))
    assert set(s) == {"ann_return", "ann_vol", "sharpe", "max_drawdown", "cvar_95_daily"}
    assert s["max_drawdown"] <= 0
