"""Grid search over GA population size, crossover rate and mutation rate.

Every configuration gets the same budget of fitness evaluations
(pop_size * generations), so a bigger population runs fewer generations. Each one
optimizes the same problems: monthly dates from the last two training years, with
trailing momentum as the return estimate. A config's score is the mean gap between
the fitness it reached and the best fitness any config reached on that date.
Only training data is used.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .ga import GAConfig, optimize
from .scenarios import monthly_scenarios

POP_SIZES = (20, 40, 80)
CROSSOVER = (0.6, 0.8, 0.95)
MUTATION = (0.02, 0.1, 0.25)
BUDGET = 4800


def tune(returns: pd.DataFrame, dates: pd.DatetimeIndex, risk_aversion: float,
         seeds: int = 2) -> pd.DataFrame:
    problems = []
    for d in dates:
        hist = returns.loc[:d]
        mu = hist.iloc[-126:].sum().values * (21 / 126)
        problems.append((mu, monthly_scenarios(returns, d)))

    rows = []
    for pop, cx, mut in itertools.product(POP_SIZES, CROSSOVER, MUTATION):
        fits = np.zeros((len(problems), seeds))
        for i, (mu, scen) in enumerate(problems):
            for s in range(seeds):
                cfg = GAConfig(pop_size=pop, generations=BUDGET // pop, crossover_rate=cx,
                               mutation_rate=mut, seed=s)
                _, hist_fit = optimize(mu, scen, risk_aversion, cfg)
                fits[i, s] = hist_fit[-1]
        rows.append({"pop_size": pop, "crossover_rate": cx, "mutation_rate": mut,
                     "fitness": fits.mean(axis=1)})
    df = pd.DataFrame(rows)
    best = np.max(np.vstack(df["fitness"].values), axis=0)
    df["gap_to_best"] = [np.mean(best - f) for f in df["fitness"]]
    df = df.drop(columns="fitness").sort_values("gap_to_best").reset_index(drop=True)
    return df
