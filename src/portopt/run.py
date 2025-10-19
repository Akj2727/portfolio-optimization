"""End-to-end experiment.

    python -m portopt.run

1. Simulate 15 years of daily returns; the last 5 years are out of sample.
2. Walk-forward RF / XGBoost / stacked forecasts of 21-day relative returns.
3. Tune the GA on the two years before the out-of-sample period.
4. Backtest, rebalancing every 21 trading days with 10 bp costs:
   Equal Weight, GA-CVaR (stacked forecasts), GA-MinCVaR (no forecasts),
   HMM-Blend and HMM-Switch.
5. Write results/summary.md, results/ga_tuning.csv and results/equity.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import run_backtest, summarize
from .data import simulate_market
from .features import aligned, cross_sectional_ic
from .ga import GAConfig, optimize
from .hmm_strategy import MarketHMM, blend_weights, regime_portfolios, switch_weights
from .models import demean_by_date, walk_forward
from .scenarios import monthly_scenarios
from .tune_ga import tune

RESULTS = Path(__file__).resolve().parents[2] / "results"
OOS_YEARS = 5
RISK_AVERSION = 1.0


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    returns, regimes = simulate_market()
    dates = returns.index
    split = dates[-252 * OOS_YEARS]
    X, y = aligned(returns)
    feat_dates = X.index.get_level_values("date").unique()
    rebal = feat_dates[feat_dates >= split][::21]
    print(f"{len(dates)} days, out-of-sample from {split.date()}, {len(rebal)} rebalances")

    # forecasts
    cache = RESULTS / "forecasts.pkl"
    if cache.exists():
        preds = pd.read_pickle(cache)
    else:
        preds = walk_forward(X, y, rebal)
        preds.to_pickle(cache)
    target = demean_by_date(y).reindex(preds.index)
    ic = {m: cross_sectional_ic(preds[m], target) for m in preds}
    ic["mom_126 (raw feature)"] = cross_sectional_ic(X.loc[preds.index, "mom_126"], target)

    # GA tuning on training data only
    tune_dates = dates[(dates < split) & (dates >= dates[-252 * (OOS_YEARS + 2)])][::21]
    grid = tune(returns, tune_dates, RISK_AVERSION)
    grid.to_csv(RESULTS / "ga_tuning.csv", index=False)
    best = grid.iloc[0]
    ga_cfg = GAConfig(pop_size=int(best.pop_size), generations=4800 // int(best.pop_size),
                      crossover_rate=float(best.crossover_rate),
                      mutation_rate=float(best.mutation_rate))

    # HMM on training market returns
    mkt = returns.mean(axis=1)
    hmm = MarketHMM().fit(mkt[mkt.index < split])

    def ga_cvar(d):
        mu = preds.xs(d, level="date")["stack"].reindex(returns.columns).values
        w, _ = optimize(mu, monthly_scenarios(returns, d), RISK_AVERSION, ga_cfg)
        return w

    def ga_min_cvar(d):
        w, _ = optimize(np.zeros(returns.shape[1]), monthly_scenarios(returns, d),
                        RISK_AVERSION, ga_cfg)
        return w

    def hmm_blend(d):
        return blend_weights(hmm.filtered_probs(mkt.loc[:d]),
                             regime_portfolios(returns.loc[:d].iloc[-126:]))

    def hmm_switch(d):
        return switch_weights(hmm.filtered_probs(mkt.loc[:d]),
                              regime_portfolios(returns.loc[:d].iloc[-126:]))

    n = returns.shape[1]
    strategies = {
        "Equal Weight": lambda d: np.full(n, 1.0 / n),
        "GA-CVaR (stacked forecasts)": ga_cvar,
        "GA-MinCVaR (no forecasts)": ga_min_cvar,
        "HMM-Blend": hmm_blend,
        "HMM-Switch": hmm_switch,
    }
    daily = {name: run_backtest(returns, rebal, fn) for name, fn in strategies.items()}
    table = pd.DataFrame({k: summarize(v) for k, v in daily.items()}).T

    write_summary(table, ic, grid, ga_cfg, split, len(rebal))
    plot(daily)
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))


def write_summary(table, ic, grid, cfg, split, n_rebal) -> None:
    fmt = table.copy()
    for c in ("ann_return", "ann_vol", "max_drawdown", "cvar_95_daily"):
        fmt[c] = (table[c] * 100).map(lambda v: f"{v:.2f}%")
    fmt["sharpe"] = table["sharpe"].map(lambda v: f"{v:.2f}")
    lines = [
        "# Results",
        "",
        f"Out-of-sample period: {split.date()} onward ({OOS_YEARS} years, {n_rebal} monthly rebalances, 10 bp costs).",
        "",
        "## Strategies",
        "",
        fmt.to_markdown(),
        "",
        "## Forecast quality (mean cross-sectional rank IC, out of sample)",
        "",
        "| Model | IC |",
        "| --- | --- |",
        *[f"| {k} | {v:.4f} |" for k, v in ic.items()],
        "",
        "## GA tuning (equal evaluation budget, training data only)",
        "",
        f"Chosen: population {cfg.pop_size}, crossover {cfg.crossover_rate}, mutation {cfg.mutation_rate}.",
        "",
        grid.head(10).to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines))


def plot(daily: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, s in daily.items():
        ax.plot((1 + s).cumprod(), label=name, linewidth=1.3)
    ax.set_ylabel("Growth of 1")
    ax.set_title("Out-of-sample equity curves")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "equity.png", dpi=130)


if __name__ == "__main__":
    main()
