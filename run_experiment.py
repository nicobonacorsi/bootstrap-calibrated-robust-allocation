from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import DEFAULT
from src.backtest import run_walk_forward
from src.data import download_fama_french_25_daily, load_local_returns
from src.metrics import summary, paired_block_bootstrap_vol_reduction_ci
from src.selection import select_hyperparameters_pre2013


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Optional local returns CSV.")
    parser.add_argument("--select", action="store_true", help="Run pre-2013 hyperparameter selection.")
    parser.add_argument("--ci-bootstrap", type=int, default=500)
    args = parser.parse_args()

    returns = load_local_returns(args.data) if args.data else download_fama_french_25_daily()
    returns = returns.loc[DEFAULT.data_start:DEFAULT.oos_end]

    cfg = DEFAULT
    out = Path("results")
    out.mkdir(exist_ok=True)

    if args.select:
        cfg, selection = select_hyperparameters_pre2013(returns, cfg)
        selection.to_csv(out / "pre2013_model_selection.csv", index=False)

    bt = run_walk_forward(returns, cfg, start=cfg.oos_start, end=cfg.oos_end)
    bt.returns.to_csv(out / "oos_daily_returns.csv")
    bt.turnover.to_csv(out / "oos_turnover.csv")
    for name, weights in bt.weights.items():
        weights.to_csv(out / f"weights_{name}.csv")

    metrics = {name: summary(bt.returns[name]) for name in bt.returns}
    ci = paired_block_bootstrap_vol_reduction_ci(
        bt.returns["robust"],
        bt.returns["equal_weight"],
        block_length=cfg.block_length,
        n_boot=args.ci_bootstrap,
        seed=cfg.random_seed,
    )
    metrics["robust_vs_equal_weight"] = {
        "relative_volatility_reduction":
            1.0 - metrics["robust"]["annualized_volatility"] /
            metrics["equal_weight"]["annualized_volatility"],
        "paired_block_bootstrap_95pct_CI": list(ci),
    }

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
