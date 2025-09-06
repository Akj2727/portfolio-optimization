import numpy as np

from portopt.risk import cvar, var


def test_cvar_matches_brute_force():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 1000)
    worst = np.sort(r)[:50]  # 5% of 1000
    assert np.isclose(cvar(r, 0.95), -worst.mean())


def test_cvar_is_at_least_var():
    rng = np.random.default_rng(1)
    r = rng.standard_t(4, 2000) * 0.01
    assert cvar(r) >= var(r)


def test_vectorised_columns_match_single():
    rng = np.random.default_rng(2)
    R = rng.normal(0, 0.01, (500, 7))
    cols = cvar(R)
    assert np.allclose(cols, [cvar(R[:, j]) for j in range(7)])
