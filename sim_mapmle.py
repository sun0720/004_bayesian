#!/usr/bin/env python3
"""MAP vs MLE arm for h001-gptsol (review 012, P1-6).

Same scalar Gaussian design and the same replications as the fixed-S arm of
sim.py.  For each replication we fit two centres on the main sample:
  - MAP: augmented posterior mode with the N(0,0.1^2) prior on delta,
    truncated to |delta| <= 6 (the implementation in sim.py);
  - MLE: the same augmented likelihood without the delta prior (flat prior),
    delta still bounded to |delta| <= 6.
We report bias, RMSE and coverage for both, so the simulation's evidence can be
compared with the theorem's augmented MLE.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from scipy.optimize import minimize

import sim

R = 500
SEED = 20261022
NS = (300, 1000, 3000)
RHO = 1.5


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
    for n in NS:
        rng = default_rng(SEED + n)
        for _ in range(R):
            y = sim.gen_data(n, rng, RHO)
            m = int(round(n ** (2.0 / 3.0)))
            m = max(10, min(m, max(10, n // 2)))
            y_main = y[m:]
            N = len(y_main)
            shift = sim.true_mean(RHO / math.sqrt(n))
            naive = float(y_main.mean())
            naive_se = float(y_main.std(ddof=1) / math.sqrt(N))
            map_c, _ = sim.fit_augmented(y_main, sim.S_STAR)
            mle_c, _ = fit_flat(y_main, sim.S_STAR)
            map_se = sim.target_se(y_main, map_c, sim.S_STAR)
            mle_se = sim.target_se(y_main, mle_c, sim.S_STAR)
            rows.append(dict(n=n, rep=_, truth=0.0, shift=shift,
                             naive=naive - shift, naive_se=naive_se,
                             map=map_c - shift, map_se=map_se,
                             mle=mle_c - shift, mle_se=mle_se))
        print(f"n={n} done ({time.time()-t0:.1f}s)", flush=True)
    with open(out / "results_mapmle.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for n in NS:
        sub = [r for r in rows if r["n"] == n]
        summary[str(n)] = {}
        for key in ("map", "mle", "naive"):
            bias = np.mean([r[key] for r in sub])
            se = np.std([r[key] for r in sub], ddof=1) / math.sqrt(len(sub))
            rmse = math.sqrt(np.mean([r[key] ** 2 for r in sub]))
            cov = np.mean([abs(r[key]) <= 1.96 * r[key + "_se"] for r in sub])
            summary[str(n)][key] = dict(bias=float(bias), mc_se=float(se),
                                        rmse=float(rmse), coverage=float(cov))
    (out / "summary_mapmle.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
