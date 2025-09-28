import numpy as np

from portopt.ga import GAConfig, fitness, optimize, project


def test_project_satisfies_constraints():
    rng = np.random.default_rng(0)
    W = project(rng.normal(0, 1, (200, 8)), 0.3)
    assert np.allclose(W.sum(axis=1), 1)
    assert (W >= -1e-12).all() and (W <= 0.3 + 1e-9).all()


def test_ga_beats_equal_weight_and_random():
    rng = np.random.default_rng(1)
    n = 8
    mu = rng.normal(0.01, 0.01, n)
    scen = rng.normal(0.0, 0.02, (500, n)) * rng.uniform(0.5, 2.0, n)
    w, hist = optimize(mu, scen, risk_aversion=0.5, cfg=GAConfig(seed=3))
    ew = np.full((1, n), 1 / n)
    rand = project(rng.dirichlet(np.ones(n), 500), 0.3)
    f_best = fitness(w[None], mu, scen, 0.5)[0]
    assert f_best >= fitness(ew, mu, scen, 0.5)[0]
    assert f_best >= fitness(rand, mu, scen, 0.5).max()
    assert np.all(np.diff(hist) >= -1e-12)  # elitism: best never gets worse


def test_ga_close_to_obvious_optimum():
    # One asset dominates on return with no extra risk: the cap should bind on it.
    n = 5
    mu = np.array([0.05, 0.0, 0.0, 0.0, 0.0])
    scen = np.random.default_rng(2).normal(0, 0.01, (400, n))
    w, _ = optimize(mu, scen, risk_aversion=0.1, cfg=GAConfig(seed=0, generations=150))
    assert abs(w[0] - 0.3) < 0.01
