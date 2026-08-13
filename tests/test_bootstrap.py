import numpy as np

from src.bootstrap import circular_block_indices, covariance_ambiguity_set


def test_circular_block_bootstrap_shape_and_range():
    rng = np.random.default_rng(4)
    idx = circular_block_indices(100, 7, rng)
    assert idx.shape == (100,)
    assert idx.min() >= 0
    assert idx.max() < 100


def test_robust_covariance_is_psd_and_inflated_in_sample_basis():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 5))
    amb = covariance_ambiguity_set(
        x, n_boot=32, block_length=10, quantile=0.90, rho=1.0, seed=10
    )
    assert np.linalg.eigvalsh(amb.robust_cov).min() > -1e-9
    robust_dir = np.einsum(
        "ij,ji->i",
        amb.eigenvectors.T @ amb.robust_cov,
        amb.eigenvectors,
    )
    assert np.all(robust_dir + 1e-12 >= amb.eigenvalues)
