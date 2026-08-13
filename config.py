from dataclasses import dataclass, field
from typing import Tuple

@dataclass(frozen=True)
class AllocationConfig:
    # Data / chronology
    data_start: str = "1995-01-01"
    selection_start: str = "2000-01-01"
    selection_end: str = "2012-12-31"
    oos_start: str = "2013-01-01"
    oos_end: str = "2026-07-31"

    # Estimation
    lookback_days: int = 756
    rebalance: str = "monthly"
    bootstrap_replications: int = 128
    block_length: int = 21
    ambiguity_quantile: float = 0.95
    robustness_rho: float = 1.0
    covariance_jitter: float = 1e-10

    # Portfolio constraints
    weight_cap: float = 0.15
    transaction_cost_bps: float = 10.0

    # Reproducibility
    random_seed: int = 20260716

    # Sensitivity / selection grid (all intended for pre-2013 use)
    rho_grid: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    cap_grid: Tuple[float, ...] = (0.08, 0.10, 0.15)
    block_grid: Tuple[int, ...] = (5, 21, 63)

DEFAULT = AllocationConfig()
