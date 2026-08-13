import numpy as np
import pandas as pd

from config import AllocationConfig
from src.backtest import run_walk_forward


def test_future_returns_do_not_change_first_oos_decision():
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2010-01-01", periods=1000)
    x = pd.DataFrame(rng.normal(scale=0.01, size=(1000, 5)), index=dates)

    cfg = AllocationConfig(
        lookback_days=252,
        oos_start=str(dates[500].date()),
        oos_end=str(dates[700].date()),
        bootstrap_replications=16,
        block_length=10,
        weight_cap=0.30,
    )

    bt1 = run_walk_forward(x, cfg, start=cfg.oos_start, end=cfg.oos_end)
    x2 = x.copy()
    x2.loc[dates[501]:] *= 20.0
    bt2 = run_walk_forward(x2, cfg, start=cfg.oos_start, end=cfg.oos_end)

    first = bt1.weights["robust"].index[0]
    np.testing.assert_allclose(
        bt1.weights["robust"].loc[first].values,
        bt2.weights["robust"].loc[first].values,
        atol=1e-12,
    )
