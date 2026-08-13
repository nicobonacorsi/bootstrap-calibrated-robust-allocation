from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class QPSolution:
    weights: np.ndarray
    objective: float
    iterations: int
    converged: bool


def _solve_equality_subproblem(
    h: np.ndarray,
    active_lower: set[int],
    active_upper: set[int],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve the QP exactly for a fixed active set."""
    n = h.shape[0]
    active = sorted(active_lower | active_upper)
    free = [i for i in range(n) if i not in active]

    w = np.zeros(n, dtype=float)
    if active_lower:
        idx = np.fromiter(sorted(active_lower), dtype=int)
        w[idx] = lower[idx]
    if active_upper:
        idx = np.fromiter(sorted(active_upper), dtype=int)
        w[idx] = upper[idx]

    remaining = 1.0 - w[active].sum() if active else 1.0
    if not free:
        if abs(remaining) > 1e-10:
            raise RuntimeError("Infeasible active set.")
        # lambda is not uniquely recoverable without a free variable.
        return w, 0.0

    f = np.asarray(free, dtype=int)
    a = np.asarray(active, dtype=int)

    h_ff = h[np.ix_(f, f)]
    linear = h[np.ix_(f, a)] @ w[a] if len(a) else np.zeros(len(f))

    kkt = np.block([
        [h_ff, np.ones((len(f), 1))],
        [np.ones((1, len(f))), np.zeros((1, 1))],
    ])
    rhs = np.concatenate([-linear, [remaining]])

    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]

    w[f] = sol[:-1]
    lam = float(sol[-1])
    return w, lam


def capped_long_only_min_variance(
    cov: np.ndarray,
    cap: float = 0.15,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> QPSolution:
    """Custom active-set solver for a capped long-only minimum-variance QP.

    Solves
        min 0.5 w' H w
        s.t. 1'w = 1, 0 <= w_i <= cap.

    The algorithm alternates between exact equality-constrained solves on the
    current free set, bound activation by a feasible line search, and KKT-based
    release of incorrectly active variables.
    """
    h = np.asarray(cov, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("cov must be a square matrix.")
    n = h.shape[0]
    if cap <= 0 or n * cap < 1.0 - tol:
        raise ValueError("cap is infeasible: n * cap must be at least 1.")

    h = 0.5 * (h + h.T)
    # Tiny ridge only for numerical stability; it does not change the intended QP.
    h = h + 1e-12 * np.eye(n)

    lower = np.zeros(n)
    upper = np.full(n, cap)

    # Equal weight is feasible whenever cap >= 1/n.
    w = np.full(n, 1.0 / n)
    active_lower: set[int] = set()
    active_upper: set[int] = set()

    for it in range(1, max_iter + 1):
        candidate, lam = _solve_equality_subproblem(
            h, active_lower, active_upper, lower, upper
        )
        free = np.array(
            [i for i in range(n) if i not in active_lower and i not in active_upper],
            dtype=int,
        )

        # If the subproblem optimum leaves the box, move to the first hit bound.
        violations = (
            (candidate[free] < lower[free] - tol)
            | (candidate[free] > upper[free] + tol)
        )
        if violations.any():
            d = candidate - w
            alpha = 1.0
            hit_idx = None
            hit_upper = False
            for i in free:
                if d[i] > tol:
                    a = (upper[i] - w[i]) / d[i]
                    if -tol <= a < alpha:
                        alpha = max(0.0, a)
                        hit_idx = int(i)
                        hit_upper = True
                elif d[i] < -tol:
                    a = (lower[i] - w[i]) / d[i]
                    if -tol <= a < alpha:
                        alpha = max(0.0, a)
                        hit_idx = int(i)
                        hit_upper = False

            if hit_idx is None:
                raise RuntimeError("Active-set line search failed to identify a bound.")
            w = w + alpha * d
            w[np.abs(w) < tol] = 0.0
            w[np.abs(w - cap) < tol] = cap
            if hit_upper:
                active_upper.add(hit_idx)
                active_lower.discard(hit_idx)
            else:
                active_lower.add(hit_idx)
                active_upper.discard(hit_idx)
            continue

        # Feasible exact optimum for the current active set.
        w = candidate
        grad = h @ w
        reduced = grad + lam

        # KKT:
        # lower-active: reduced_i >= 0
        # upper-active: reduced_i <= 0
        lower_bad = [(i, reduced[i]) for i in active_lower if reduced[i] < -tol]
        upper_bad = [(i, reduced[i]) for i in active_upper if reduced[i] > tol]

        if not lower_bad and not upper_bad:
            w = np.clip(w, 0.0, cap)
            # Correct microscopic budget error while preserving feasibility.
            budget_error = 1.0 - w.sum()
            if abs(budget_error) > 1e-12:
                free_room = np.where((w > tol) & (w < cap - tol))[0]
                if len(free_room):
                    w[free_room[0]] += budget_error
            obj = float(w @ h @ w)
            return QPSolution(w, obj, it, True)

        # Release the most severe KKT violation.
        candidates = [(abs(v), i, "L") for i, v in lower_bad]
        candidates += [(abs(v), i, "U") for i, v in upper_bad]
        _, idx, side = max(candidates)
        if side == "L":
            active_lower.remove(idx)
        else:
            active_upper.remove(idx)

    obj = float(w @ h @ w)
    return QPSolution(w, obj, max_iter, False)
