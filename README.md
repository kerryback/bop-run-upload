# The Virtue of Complexity in Simple Economic Models

Code for Back, Ober and Pruitt, *"The Virtue of Complexity in Simple Economic Models"*.

Three structural asset-pricing models are simulated to a firm panel, and in each one the **true**
stochastic discount factor is known. That makes it possible to ask a question the empirical
literature cannot: when a high-complexity estimator beats a linear one out of sample, how much of
the available prize did it actually collect, and how much was there to collect in the first place?

The estimators compared are **DKKM** (random Fourier features plus ridge, in the Didisheim–Ke–Kelly–
Malamud sense) against a **fair linear benchmark** — Fama-French, Fama-MacBeth, linear-in-ranks and
the market, each given the same conditioning information and the same evaluation.

| model | reference |
|---|---|
| **BGN** | Berk, Green and Naik (1999) |
| **KP14** | Kogan and Papanikolaou (2014) |
| **GS21** | Gomes and Schmid (2021) |

## Start here

**[`docs/quickstart.md`](docs/quickstart.md)** — what the research documents hold, the measurement
protocol, and the loop a new economy goes through from proposal to graded result.

| document | what it holds |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | every economy's numbers, and the numbered cross-cutting findings |
| [`docs/NEXTUP.md`](docs/NEXTUP.md) | the live queue, and every experiment already run with its pre-registration and grade |
| [`docs/RUNS.md`](docs/RUNS.md) | cluster procedure, hazards, and the measured cost of every campaign |
| [`variants/README.md`](variants/README.md) | the pipeline itself: economies, estimators, feature bases, and the solve registry |

## Layout

```
variants/            THE LIVE PIPELINE. Engineered economies, the oracle and the estimators.
  common/              protocol.py (the measurement protocol), oracle.py (feature bases and the
                       two-pass population evaluation), solstamp.py (content-addressed solves)
  bgn_gam/ kp_vy/ gs_bx/   per-model solvers, simulators and parameters
  run_oracle.py        simulate a panel, compute true conditional moments, report every ceiling
  run_estimators.py    DKKM and the linear benchmarks on rolling windows, scored on truth
  aggregate_seeds.py   the results table; refuses to write an off-protocol row
  penalty_gate.py      did the ridge grid bind? Run before reading any number

experiments/
  specs/               one JSON per economy: the override, the registered prediction, the
                       falsifier, and the solve ids precommitted BEFORE solving
  registry/            one manifest per solve, committed; outlives the artifact it describes

utils_bgn/ utils_kp14/ utils_gs21/ utils_factors/
                     the published-model implementations, and the reference the variants are
                     held to
config.py            the parameter authority. tests/test_config_parity.py asserts the variants
                     have not drifted from it
tests/               275 tests. They pin the protocol, the precommitments, the docs against the
                     data, and the hazards that have cost money
docs/                the research record (above)
legacy/              the superseded 7-step pipeline. Produces no reportable number
archive/             retired diagnostics, notes and validation work
voc_diagnosis/       the 2026-08 audit that established the K-factor ceiling
```

## Install

```bash
conda env create -f environment.yml && conda activate bop
# or
pip install -r requirements.txt pyarrow
```

Then `python -m pytest tests/ -q` — expect 275 passed.

## How the method works, in one pass

1. **Solve the model.** Each parametrization needs its own solve: the exposures, the price of the
   priced factor, its persistence and the regime multipliers all enter the value functions. Solves
   are content-addressed — same parameters and same source means the same `solve_id`, so an
   unchanged spec exits immediately instead of re-solving. Every solve writes a manifest to
   `experiments/registry/`, which is committed and outlives the artifact.
2. **Simulate and take the truth.** `run_oracle.py` draws a panel and computes the true conditional
   mean and covariance every month. From those it reports, for each feature basis, the conditional
   oracle and the best constant-coefficient population portfolio. The gap between the nonlinear and
   linear population ceilings is **room**: what a nonlinear method could win if it estimated
   perfectly.
3. **Estimate.** `run_estimators.py` runs DKKM and the linear benchmarks on rolling 360-month
   windows and scores every resulting portfolio against the true moments, so there is no sampling
   error in the evaluation — only in the estimation.
4. **Compare.** The headline is the **fair gap**: DKKM minus the best fair linear method. It
   decomposes exactly as `gap = room − excess estimation loss`, which is the identity most of the
   findings are about.

## Where the result currently stands

Of sixteen economies at one protocol, **two have a fair gap distinguishable from zero**: `vyg25`
(+0.0148, t 5.2) and `g0235f` (+0.0079, t 4.8). Both are loading-shape results at a fixed number of
priced shocks. Three levers have been tested and closed — the number of priced shocks, estimation
density, and the width of the cross-section — each by a pre-registered experiment that came back
against the proposal.

`docs/RESULTS.md` carries all of it, including what was withdrawn: a solve-level pricing defect
found on 2026-09-18 took the project's original headline with it, and no pre-fix figure is reported
as a result.
