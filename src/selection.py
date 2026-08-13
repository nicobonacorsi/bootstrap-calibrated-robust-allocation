from __future__ import annotations

from dataclasses import replace
from itertools import product
import pandas as pd

from config import AllocationConfig
from .backtest import run_walk_forward
from .metrics import annualized_volatility


def select_hyperparameters_pre2013(
    returns: pd.DataFrame,
    cfg: AllocationConfig,
    eval_start: str = "2008-01-01",
) -> tuple[AllocationConfig, pd.DataFrame]:
    """Select rho/cap/block length using only the pre-2013 development period.

    This intentionally never reads the 2013+ confirmation sample.
    The selection objective is realized annualized volatility.
    """
    if pd.Timestamp(cfg.selection_end) >= pd.Timestamp(cfg.oos_start):
        raise ValueError("selection_end must precede oos_start.")

    rows = []
    for rho, cap, block in product(cfg.rho_grid, cfg.cap_grid, cfg.block_grid):
        candidate = replace(
            cfg,
            robustness_rho=rho,
            weight_cap=cap,
            block_length=block,
        )
        bt = run_walk_forward(
            returns.loc[:cfg.selection_end],
            candidate,
            start=max(pd.Timestamp(eval_start), pd.Timestamp(cfg.selection_start)).strftime("%Y-%m-%d"),
            end=cfg.selection_end,
        )
        vol = annualized_volatility(bt.returns["robust"])
        rows.append({"rho": rho, "cap": cap, "block_length": block, "ann_vol": vol})

    table = pd.DataFrame(rows).sort_values("ann_vol").reset_index(drop=True)
    best = table.iloc[0]
    selected = replace(
        cfg,
        robustness_rho=float(best["rho"]),
        weight_cap=float(best["cap"]),
        block_length=int(best["block_length"]),
    )
    return selected, table
