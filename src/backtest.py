from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import AllocationConfig
from .bootstrap import covariance_ambiguity_set
from .optimizer import capped_long_only_min_variance


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    turnover: pd.DataFrame


def _rebalance_dates(index: pd.DatetimeIndex, mode: str = "monthly") -> set[pd.Timestamp]:
    if mode != "monthly":
        raise NotImplementedError("Only monthly rebalancing is implemented.")
    s = pd.Series(np.arange(len(index)), index=index)
    dates = s.groupby(index.to_period("M")).head(1).index
    return set(pd.Timestamp(d) for d in dates)


def _drift_weights(w: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    gross = w * (1.0 + asset_returns)
    denom = gross.sum()
    if denom <= 0:
        return w.copy()
    return gross / denom


def _trade_cost(current: np.ndarray, target: np.ndarray, bps: float) -> tuple[float, float]:
    traded_notional = float(np.abs(target - current).sum())
    cost = traded_notional * bps * 1e-4
    return traded_notional, cost


def run_walk_forward(
    returns: pd.DataFrame,
    cfg: AllocationConfig,
    start: str | None = None,
    end: str | None = None,
) -> BacktestResult:
    """Walk-forward robust, equal-weight, and Ledoit-Wolf portfolios.

    Every covariance estimate at date t uses rows strictly before t.
    Portfolio holdings drift between monthly rebalances. Transaction costs are
    deducted on rebalance dates as bps times total one-way traded notional
    sum_i |w_i^target - w_i^pretrade|.
    """
    x = returns.sort_index().copy()
    p = x.shape[1]
    if p * cfg.weight_cap < 1 - 1e-12:
        raise ValueError("weight_cap is infeasible for the number of assets.")

    start_ts = pd.Timestamp(start or cfg.oos_start)
    end_ts = pd.Timestamp(end or cfg.oos_end)
    eval_index = x.loc[start_ts:end_ts].index
    rebal = _rebalance_dates(eval_index, cfg.rebalance)

    names = ["robust", "equal_weight", "ledoit_wolf"]
    current = {name: np.full(p, 1.0 / p) for name in names}
    ret_rows = []
    turnover_rows = []
    weight_records = {name: [] for name in names}

    for t in eval_index:
        loc = x.index.get_loc(t)
        if isinstance(loc, slice):
            raise ValueError("Return index must be unique.")
        hist = x.iloc[max(0, loc - cfg.lookback_days):loc]

        costs = {name: 0.0 for name in names}
        turns = {name: 0.0 for name in names}

        if t in rebal and len(hist) >= max(126, cfg.block_length + 2):
            seed = cfg.random_seed + int(pd.Timestamp(t).strftime("%Y%m%d"))

            amb = covariance_ambiguity_set(
                hist.to_numpy(),
                n_boot=cfg.bootstrap_replications,
                block_length=cfg.block_length,
                quantile=cfg.ambiguity_quantile,
                rho=cfg.robustness_rho,
                seed=seed,
                jitter=cfg.covariance_jitter,
            )
            robust_target = capped_long_only_min_variance(
                amb.robust_cov, cfg.weight_cap
            ).weights

            lw_cov = LedoitWolf().fit(hist.to_numpy()).covariance_
            lw_target = capped_long_only_min_variance(
                lw_cov, cfg.weight_cap
            ).weights

            targets = {
                "robust": robust_target,
                "equal_weight": np.full(p, 1.0 / p),
                "ledoit_wolf": lw_target,
            }
            for name in names:
                turns[name], costs[name] = _trade_cost(
                    current[name], targets[name], cfg.transaction_cost_bps
                )
                current[name] = targets[name].copy()

        day = x.loc[t].to_numpy(dtype=float)
        row = {}
        for name in names:
            gross_ret = float(current[name] @ day)
            row[name] = gross_ret - costs[name]
            weight_records[name].append((t, *current[name]))
            current[name] = _drift_weights(current[name], day)

        ret_rows.append((t, row["robust"], row["equal_weight"], row["ledoit_wolf"]))
        turnover_rows.append((t, turns["robust"], turns["equal_weight"], turns["ledoit_wolf"]))

    ret_df = pd.DataFrame(
        ret_rows, columns=["date", "robust", "equal_weight", "ledoit_wolf"]
    ).set_index("date")
    turn_df = pd.DataFrame(
        turnover_rows, columns=["date", "robust", "equal_weight", "ledoit_wolf"]
    ).set_index("date")

    cols = ["date"] + list(x.columns)
    weight_dfs = {
        name: pd.DataFrame(records, columns=cols).set_index("date")
        for name, records in weight_records.items()
    }
    return BacktestResult(ret_df, weight_dfs, turn_df)
