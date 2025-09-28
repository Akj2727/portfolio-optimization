"""Return forecasting: Random Forest, XGBoost and a stacked ensemble, fit walk-forward.

Training only ever uses rows whose 21-day target window has fully closed before the
forecast date (a purge gap of HORIZON days), so no future information leaks in.
Models are refit once a year on an expanding window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .features import HORIZON


def build_models(seed: int = 0) -> dict:
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=50,
                               max_features=0.5, n_jobs=2, random_state=seed)
    xgb = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                       colsample_bytree=0.8, reg_lambda=5.0, n_jobs=2, random_state=seed)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    stack = StackingRegressor(
        estimators=[("rf", rf), ("xgb", xgb), ("ridge", ridge)],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 2, 12)),
        cv=KFold(5, shuffle=False),  # contiguous time blocks
        n_jobs=1,
    )
    return {"rf": rf, "xgb": xgb, "stack": stack}


def demean_by_date(y: pd.Series) -> pd.Series:
    """Cross-sectional target: return relative to that day's average across assets."""
    return y - y.groupby(level="date").transform("mean")


def walk_forward(X: pd.DataFrame, y: pd.Series, forecast_dates: pd.DatetimeIndex,
                 refit_every: int = 252, seed: int = 0) -> pd.DataFrame:
    """Forecasts for every (date, asset) in forecast_dates, one column per model."""
    all_dates = X.index.get_level_values("date").unique().sort_values()
    target = demean_by_date(y)
    out, models, last_fit = [], None, None
    for d in forecast_dates:
        pos = all_dates.get_loc(d)
        if models is None or pos - last_fit >= refit_every:
            cutoff = all_dates[pos - HORIZON]  # targets after this would see past d
            train = X.index.get_level_values("date") <= cutoff
            models = build_models(seed)
            for m in models.values():
                m.fit(X[train].values, target[train].values)
            last_fit = pos
        rows = X.xs(d, level="date", drop_level=False)
        preds = {name: m.predict(rows.values) for name, m in models.items()}
        out.append(pd.DataFrame(preds, index=rows.index))
    return pd.concat(out)
