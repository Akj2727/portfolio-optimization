"""Genetic algorithm for long-only portfolio weights under a CVaR penalty.

Objective (maximised):  w . mu  -  risk_aversion * CVaR_95(scenarios @ w)
Constraints:            w >= 0,  sum(w) = 1,  w_i <= max_weight

Selection is by tournament, crossover is a random convex blend of two parents,
mutation adds Gaussian noise to a random subset of genes, and every child is
repaired back onto the constraint set. The best `elite` portfolios survive unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .risk import ALPHA, cvar


@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 60
    generations: int = 80
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1
    tournament: int = 3
    elite: int = 2
    max_weight: float = 0.30
    seed: int = 0


def project(w: np.ndarray, max_weight: float) -> np.ndarray:
    """Maps rows onto {w >= 0, sum = 1, w <= max_weight} by clipping and redistributing."""
    w = np.clip(np.atleast_2d(w).astype(float), 0.0, None)
    n = w.shape[1]
    if max_weight * n < 1 - 1e-12:
        raise ValueError("max_weight too small for the number of assets")
    s = w.sum(axis=1, keepdims=True)
    w = np.where(s > 0, w / np.where(s > 0, s, 1), 1.0 / n)
    for _ in range(50):
        over = w > max_weight
        if not over.any():
            break
        excess = np.where(over, w - max_weight, 0).sum(axis=1, keepdims=True)
        w = np.where(over, max_weight, w)
        room = np.where(over, 0, max_weight - w)
        room_sum = room.sum(axis=1, keepdims=True)
        w = w + np.where(room_sum > 0, excess * room / np.where(room_sum > 0, room_sum, 1), 0)
    return w


def fitness(W: np.ndarray, mu: np.ndarray, scenarios: np.ndarray, risk_aversion: float,
            alpha: float = ALPHA) -> np.ndarray:
    """Objective for every row of W at once."""
    return W @ mu - risk_aversion * cvar(scenarios @ W.T, alpha)


def optimize(mu: np.ndarray, scenarios: np.ndarray, risk_aversion: float,
             cfg: GAConfig = GAConfig()) -> tuple[np.ndarray, list[float]]:
    """Returns (best weights, best fitness per generation)."""
    rng = np.random.default_rng(cfg.seed)
    n = len(mu)
    pop = project(rng.dirichlet(np.ones(n), cfg.pop_size), cfg.max_weight)
    pop[0] = project(np.full(n, 1.0 / n), cfg.max_weight)[0]  # seed with equal weight
    fit = fitness(pop, mu, scenarios, risk_aversion)
    history = [float(fit.max())]

    for _ in range(cfg.generations):
        order = np.argsort(-fit)
        children = [pop[i] for i in order[: cfg.elite]]
        while len(children) < cfg.pop_size:
            a = _tournament(rng, fit, cfg.tournament)
            b = _tournament(rng, fit, cfg.tournament)
            if rng.random() < cfg.crossover_rate:
                lam = rng.random(n)
                child = lam * pop[a] + (1 - lam) * pop[b]
            else:
                child = pop[a].copy()
            mask = rng.random(n) < cfg.mutation_rate
            child = child + mask * rng.normal(0.0, cfg.mutation_scale / n, n)
            children.append(child)
        pop = project(np.array(children), cfg.max_weight)
        fit = fitness(pop, mu, scenarios, risk_aversion)
        history.append(float(fit.max()))

    return pop[int(np.argmax(fit))], history


def _tournament(rng: np.random.Generator, fit: np.ndarray, k: int) -> int:
    idx = rng.integers(0, len(fit), k)
    return int(idx[np.argmax(fit[idx])])
