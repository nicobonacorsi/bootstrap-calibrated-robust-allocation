# Bootstrap-Calibrated Distributionally Robust Allocation

**Nicolò Bonacorsi — Independent Quantitative Research**

A walk-forward portfolio-allocation study on the 25 Fama–French size–value
portfolios. The core idea is to estimate **covariance uncertainty**, not just a
single covariance matrix, using a circular block bootstrap; convert that
sampling uncertainty into an eigen-direction ambiguity set; and solve the
resulting capped long-only robust minimum-variance problem with a **custom
active-set QP solver**.

## Core formulation

For the sample covariance
\[
\widehat\Sigma = Q\Lambda Q^\top,
\]
each circular-block bootstrap sample \(b\) yields
\(\widehat\Sigma^{(b)}\). In the *sample* eigenbasis, define
\[
z_{bk}=q_k^\top\widehat\Sigma^{(b)}q_k.
\]
The directional uncertainty radius is
\[
r_k = Q_\alpha\!\left(\left|z_{bk}-\lambda_k\right|\right),
\]
and the robust covariance used by the optimizer is
\[
\Sigma_{\rm rob}
=
Q\,{\rm diag}(\lambda_k+\rho r_k)\,Q^\top.
\]

The portfolio problem is
\[
\min_w\; w^\top\Sigma_{\rm rob}w
\quad\text{s.t.}\quad
\mathbf 1^\top w=1,\qquad 0\le w_i\le c.
\]

`src/optimizer.py` implements the KKT active-set algorithm directly: exact
equality-constrained solves on the current free set, feasible line search to a
newly hit bound, and KKT-based release of incorrectly active coordinates.

## Experimental discipline

- **Universe:** 25 U.S. portfolios formed on Size and Book-to-Market.
- **Dependence-aware uncertainty:** circular moving-block bootstrap.
- **Primary bootstrap size:** 128 replications per rebalance.
- **Block-length sensitivity:** 5 / 21 / 63 trading days.
- **Model selection:** restricted to **2000–2012**.
- **Confirmation period:** **2013–2026**, walk-forward.
- **Constraints:** long-only, fully invested, per-asset cap.
- **Costs:** explicit weight drift and **10 bps** transaction costs.
- **Benchmarks:** equal weight and Ledoit–Wolf minimum variance.
- **Inference:** paired circular-block bootstrap CI for the relative volatility
  reduction.

No covariance estimate at date \(t\) is allowed to use returns from \(t\) or
later.

## Archived research-run results

The archived run reported:

| Metric | Robust allocation | Benchmark / comparison |
|---|---:|---:|
| Annualized volatility | **16.45%** | 20.15% equal weight |
| Relative volatility reduction | **18.4%** | 95% paired block-bootstrap CI: 15.0%–22.6% |
| CAGR | **12.90%** | — |
| Net Sharpe | **0.82** | after 10 bps costs |
| Ledoit–Wolf min-vol | 16.46% | essentially matched |

These reference values are stored in `results/reference_results.json`. Exact
reruns can differ if an upstream historical data vintage is revised or if a
different pre-2013 validation specification is selected.

## Repository structure

```text
.
├── config.py
├── run_experiment.py
├── src/
│   ├── data.py
│   ├── bootstrap.py
│   ├── optimizer.py
│   ├── backtest.py
│   ├── metrics.py
│   └── selection.py
├── tests/
├── data/
└── results/
```

## Run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the current default specification
python run_experiment.py

# Re-run the pre-2013 rho/cap/block-length selection first
python run_experiment.py --select
```

## Tests

```bash
pytest -q
```

The test suite checks the circular bootstrap, positive-semidefinite robust
covariance construction, the custom active-set solver against SciPy SLSQP, and
a direct no-look-ahead invariant.

## Research note

The purpose of the ambiguity set is not to claim that the bootstrap covariance
matrices are the true distribution. It uses their dispersion as a
data-dependent estimate of which covariance directions are unstable, then
penalizes those directions explicitly inside the allocation problem.

This repository is research code, not investment advice.
