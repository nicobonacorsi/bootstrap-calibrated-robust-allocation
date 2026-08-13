from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AmbiguityEstimate:
    sample_cov: np.ndarray
    robust_cov: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    directional_radii: np.ndarray
    bootstrap_directional_variances: np.ndarray


def circular_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n indices using a circular moving-block bootstrap."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 1 <= block_length <= n:
        raise ValueError("block_length must satisfy 1 <= block_length <= n.")

    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_length)
    idx = (starts[:, None] + offsets[None, :]) % n
    return idx.ravel()[:n]


def covariance_ambiguity_set(
    returns: np.ndarray,
    n_boot: int = 128,
    block_length: int = 21,
    quantile: float = 0.95,
    rho: float = 1.0,
    seed: int = 0,
    jitter: float = 1e-10,
) -> AmbiguityEstimate:
    """Bootstrap-calibrated, eigenvector-preserving covariance ambiguity estimate.

    Let S = Q diag(lambda) Q'. For each bootstrap covariance S_b, evaluate
    z_{b,k} = q_k' S_b q_k. The directional radius is the chosen quantile of
    |z_{b,k} - lambda_k|. The worst-case diagonal member in this directional
    ambiguity family is represented by lambda_k + rho * radius_k.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("returns must be a 2D array with at least 3 rows.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1).")
    if rho < 0:
        raise ValueError("rho must be non-negative.")

    sample_cov = np.cov(x, rowvar=False, ddof=1)
    sample_cov = 0.5 * (sample_cov + sample_cov.T)
    p = sample_cov.shape[0]
    sample_cov = sample_cov + jitter * np.eye(p)

    eigvals, eigvecs = np.linalg.eigh(sample_cov)
    eigvals = np.maximum(eigvals, jitter)

    rng = np.random.default_rng(seed)
    directional = np.empty((n_boot, p), dtype=float)

    for b in range(n_boot):
        idx = circular_block_indices(x.shape[0], block_length, rng)
        sb = np.cov(x[idx], rowvar=False, ddof=1)
        sb = 0.5 * (sb + sb.T)
        # diag(Q' S_b Q), computed without materializing every diagonal matrix.
        directional[b] = np.einsum("ij,ji->i", eigvecs.T @ sb, eigvecs)

    radii = np.quantile(np.abs(directional - eigvals[None, :]), quantile, axis=0)
    robust_eigvals = np.maximum(eigvals + rho * radii, jitter)
    robust_cov = (eigvecs * robust_eigvals) @ eigvecs.T
    robust_cov = 0.5 * (robust_cov + robust_cov.T)

    return AmbiguityEstimate(
        sample_cov=sample_cov,
        robust_cov=robust_cov,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        directional_radii=radii,
        bootstrap_directional_variances=directional,
    )
