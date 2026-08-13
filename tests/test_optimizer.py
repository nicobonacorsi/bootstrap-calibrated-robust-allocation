import numpy as np
from scipy.optimize import minimize

from src.optimizer import capped_long_only_min_variance


def test_custom_active_set_matches_slsqp():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(8, 8))
    h = a.T @ a + 0.2 * np.eye(8)
    cap = 0.25

    ours = capped_long_only_min_variance(h, cap=cap)
    assert ours.converged
    assert abs(ours.weights.sum() - 1.0) < 1e-8
    assert ours.weights.min() >= -1e-9
    assert ours.weights.max() <= cap + 1e-9

    fun = lambda w: float(w @ h @ w)
    ref = minimize(
        fun,
        np.full(8, 1/8),
        method="SLSQP",
        bounds=[(0.0, cap)] * 8,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    assert ref.success
    assert abs(fun(ours.weights) - fun(ref.x)) < 1e-7
