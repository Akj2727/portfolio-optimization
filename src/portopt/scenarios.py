"""Scenario sets for CVaR: overlapping monthly returns from a trailing window."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import HORIZON


def monthly_scenarios(returns: pd.DataFrame, asof: pd.Timestamp, lookback: int = 504,
                      horizon: int = HORIZON) -> np.ndarray:
    """Overlapping `horizon`-day returns over the `lookback` days up to and including asof."""
    hist = returns.loc[:asof].iloc[-lookback:]
    return hist.rolling(horizon).sum().dropna().values
