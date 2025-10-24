import numpy as np
import pandas as pd

from portopt.data import simulate_market, MarketConfig
from portopt.features import make_features, make_target


def test_features_do_not_change_when_future_changes():
    r, _ = simulate_market(MarketConfig(years=2, seed=3))
    cut = r.index[300]
    X1 = make_features(r)
    r2 = r.copy()
    r2.loc[r2.index > cut] = 0.05  # rewrite the future
    X2 = make_features(r2)
    a = X1.loc[X1.index.get_level_values("date") <= cut]
    b = X2.loc[X2.index.get_level_values("date") <= cut]
    pd.testing.assert_frame_equal(a, b)


def test_target_is_next_21_days():
    r, _ = simulate_market(MarketConfig(years=1, seed=4))
    y = make_target(r, 21)
    d = r.index[100]
    expected = r.iloc[101:122]["A00"].sum()
    assert np.isclose(y.loc[(d, "A00")], expected)


def test_simulation_is_reproducible():
    a, _ = simulate_market(MarketConfig(years=1, seed=5))
    b, _ = simulate_market(MarketConfig(years=1, seed=5))
    pd.testing.assert_frame_equal(a, b)
