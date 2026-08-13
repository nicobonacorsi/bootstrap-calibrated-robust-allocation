from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_volatility(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    wealth = float((1.0 + r).prod())
    years = len(r) / TRADING_DAYS
    return wealth ** (1.0 / years) - 1.0


def sharpe(r: pd.Series, daily_rf: float = 0.0) -> float:
    x = r.dropna() - daily_rf
    sd = x.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(np.sqrt(TRADING_DAYS) * x.mean() / sd)


def max_drawdown(r: pd.Series) -> float:
    wealth = (1.0 + r.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(-drawdown.min())


def summary(r: pd.Series) -> dict[str, float]:
    return {
        "CAGR": cagr(r),
        "annualized_volatility": annualized_volatility(r),
        "Sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
    }


def paired_block_bootstrap_vol_reduction_ci(
    strategy: pd.Series,
    benchmark: pd.Series,
    block_length: int = 21,
    n_boot: int = 2000,
    seed: int = 123,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Paired circular-block CI for 1 - vol(strategy)/vol(benchmark)."""
    from .bootstrap import circular_block_indices

    pair = pd.concat([strategy, benchmark], axis=1).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = circular_block_indices(len(pair), block_length, rng)
        s, e = pair[idx, 0], pair[idx, 1]
        stats[b] = 1.0 - np.std(s, ddof=1) / np.std(e, ddof=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
