# Replication package

- **Repository:** https://github.com/sun0720/004_bayesian
- **DOI:** [CONFIRM] archive DOI once the release is minted
- **License:** MIT — see `LICENSE`
- **Environment:** Python 3.11; `requirements.txt` is authoritative where the description below is less specific

## Contents

- **Scripts (entry points):** `sim.py`, `sim_drift.py`, `sim_mapmle.py`, `sim_mle_grid.py`, `sim_nuisance.py`, `sim_nuisance_ls.py`, `sim_p10.py`, `sim_p10_mle.py`, `sim_projection.py`
- **Outputs:** `results/`

## Package description

**Paper:** Finite-Threshold Screening Bounds for Locally Robust Bayesian Inference with a Bounded Dictionary

This package reproduces every number and figure in the manuscript. It uses simulated data only.

## Environment

- Python 3.11.15
- numpy 2.4.6
- scipy 1.17.1
- matplotlib 3.11.2

Install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Files

| File | What it does | Output |
|---|---|---|
| `sim.py` | Main design: Tables 4–7 and Figures 1–3 | `results/results.csv`, `results/results_boundary.csv`, `results/summary.json`, `figures/*.png` |
| `sim_mapmle.py` | MAP vs unpenalised MLE, short grid (superseded by `sim_mle_grid.py`) | `results/results_mapmle.csv` |
| `sim_mle_grid.py` | Full-grid MAP vs MLE, Table 10 | `results/results_mle_grid.csv`, `results/summary_mle_grid.json` |
| `sim_p10.py` | Two-direction P10 arm, Table 11 (original run) | `results/results_p10.csv` |
| `sim_p10_mle.py` | Two-direction P10 arm with MAP and MLE, Table 11 | `results/results_p10_mle.csv`, `results/summary_p10_mle.json` |
| `sim_drift.py` | Drifting-sensitivity arm, Table 9 | `results/results_drift.csv`, `results/summary_drift.json` |
| `sim_nuisance.py` | One-nuisance arm, Table 8 | `results/results_nuisance.csv`, `results/summary_nuisance.json` |
| `sim_nuisance_ls.py` | Least-squares variant of the nuisance arm | `results/results_nuisance_ls.csv` |
| `sim_projection.py` | Non-vacuous projection arm, Table 12 | `results/results_projection.csv`, `results/summary_projection.json` |
| `plan.md` | Simulation design notes | — |

## Reproduction commands

```bash
# Table 4 and Figures 1–3 (main design)
python3 sim.py --R 1000 --seed 20261020

# Table 10 (full-grid MAP vs MLE)
python3 sim_mle_grid.py

# Table 11 (two-direction P10 arm, MAP and MLE)
python3 sim_p10_mle.py

# Table 9 (drifting sensitivity)
python3 sim_drift.py

# Table 8 (one nuisance parameter)
python3 sim_nuisance.py

# Table 12 (non-vacuous projection)
python3 sim_projection.py
```

Run each script from inside this `replication/` directory. The scripts write into `results/`; the manuscript's `results_extra.csv` is the consolidated table used by the number checker and can be regenerated from the summaries.

## Seeds and grids

- Main design: seed 20261020, R = 1000, n ∈ {100, 300, 1000, 3000, 10000}, ρ ∈ {0, 1.5}.
- MAP/MLE full grid: seed 20261020, R = 1000, same streams as the main design.
- Two-direction arm: seed 20261024, R = 500.
- Drifting arm: seed 20261023, R = 500.
- Nuisance arm: seed 20261022, R = 500.
- Projection arm: seed 20261025, R = 200.

## Notes

- `results_extra.csv` is a consolidated summary for Tables 10–12; it is derived from the raw result files, not a separate simulation.
- Figures in `figures/` are the copies used in the manuscript; the plotting code is inside `sim.py`.
- The manuscript reports raw bias relative to γ0 = 0. Earlier internal runs that subtracted the true tilt mean are not part of this package; the current tables use the raw normalisation.

## Not included

Internal working material stays in the project repository; it is not part of this archive.

- `plan.md` — not a script, results file, or README; internal design note (DGP planning, seeds); keep it in the project repository
- `requirements.txt` — not a script, results file, or README
- `figures/rate_coverage_rho1.5.png` — not part of this archive
- `figures/selection_rho1.5.png` — not part of this archive

## Citation

If you use this material, cite the paper and this archive ([DOI to be added]).
