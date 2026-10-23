#!/usr/bin/env python3
"""Full-grid MAP vs unpenalised MLE arm for h001-gptsol (review 013, P1-6).

Same DGP, splits and common random streams as the main fixed-S arm of
sim.py: for rho = 1.5 the R streams are spawned once and reused across the
five sample sizes.  For each replication we fit
  - MAP: augmented posterior mode with the N(0, 0.1^2) prior on delta,
    truncated to |delta| <= 6 (the implementation of sim.py);
  - MLE: the same augmented likelihood with a flat delta prior on [-6, 6].
Bias is reported after subtracting the true mean of the tilted density, so
the columns are directly comparable with Table 10 of the manuscript.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng, SeedSequence
from scipy.optimize import minimize

import sim

R = 1000
SEED = 20261020
RHO = 1.5
NS = (100, 300, 1000, 3000, 10000)


def neg_loglik_flat(params, y, S, N):
    gamma, delta = params
    e = y - gamma
    if len(S) == 0:
        return 0.5 * np.sum(e * e) + 0.5 * N * math.log(2.0 * math.pi)
    r = sim.r_of(e, S)
    kz = sim.k0(delta * r)
    if not np.isfinite(kz).all() or np.any(kz <= 0.0):
        return 1e12
    Ez = sim.normal_expect_k0(delta, S)
    if not np.isfinite(Ez) or Ez <= 0.0:
        return 1e12
    return -( -0.5 * np.sum(e * e) - 0.5 * N * math.log(2.0 * math.pi)
              + np.sum(np.log(kz)) - N * math.log(Ez) )


def fit_flat(y, S):
    N = len(y)
    x0 = np.array([float(y.mean()), 0.0])
    res = minimize(neg_loglik_flat, x0, args=(y, S, N), method="L-BFGS-B",
                   bounds=[(-np.inf, np.inf), (-6.0, 6.0)],
                   options={"maxiter": 150, "ftol": 1e-10})
    if not res.success or not np.isfinite(res.fun):
        return float(y.mean()), 0.0
    return float(res.x[0]), float(res.x[1])


def main() -> int:
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    rows = []
    t0 = time.time()
    streams = SeedSequence(SEED).spawn(R)
    for n in NS:
        for rep, stream in enumerate(streams):
            rng = default_rng(stream)
            y = sim.gen_data(n, rng, RHO)
            m = int(round(n ** (2.0 / 3.0)))
            m = max(10, min(m, max(10, n // 2)))
            y_main = y[m:]
            N = len(y_main)
            shift = sim.true_mean(RHO / math.sqrt(n))
            naive = float(y_main.mean()) - shift
            naive_se = float(y_main.std(ddof=1) / math.sqrt(N))
            map_c, _ = sim.fit_augmented(y_main, sim.S_STAR)
            mle_c, _ = fit_flat(y_main, sim.S_STAR)
            map_se = sim.target_se(y_main, map_c, sim.S_STAR)
            mle_se = sim.target_se(y_main, mle_c, sim.S_STAR)
            rows.append(dict(n=n, rep=rep, truth=0.0, shift=shift,
                             naive=naive, naive_se=naive_se,
                             map=map_c - shift, map_se=map_se,
                             mle=mle_c - shift, mle_se=mle_se))
        print(f"n={n} done ({time.time()-t0:.1f}s)", flush=True)
    with open(out / "results_mle_grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for n in NS:
        sub = [r for r in rows if r["n"] == n]
        summary[str(n)] = {}
        for key in ("map", "mle", "naive"):
            vals = np.array([r[key] for r in sub], dtype=float)
            ses = np.array([r[key + "_se"] for r in sub], dtype=float)
            bias = float(vals.mean())
            mc_se = float(vals.std(ddof=1) / math.sqrt(len(vals)))
            rmse = float(math.sqrt(np.mean(vals ** 2)))
            cov = float(np.mean(np.abs(vals) <= 1.96 * ses))
            summary[str(n)][key] = dict(bias=bias, mc_se=mc_se, rmse=rmse,
                                        coverage=cov)
    (out / "summary_mle_grid.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
