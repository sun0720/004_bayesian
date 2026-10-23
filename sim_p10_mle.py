#!/usr/bin/env python3
"""MAP vs unpenalised MLE in the non-degenerate P10 arm (review 013, P1-7).

Same design, seed and replications as sim_p10.py.  For each replication we
fit the selected pair with the N(0, 0.1^2) prior on the scalar delta (MAP)
and with a flat delta prior on [-6, 6] (MLE), so the coverage gap can be
attributed to the centre rather than to the dimension of delta.  The target
is scalar (k = 1), so a scalar delta is the construction of P10, not a
restriction.
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

import sim_p10 as base

R = base.R
SEED = base.SEED
NS = base.NS
A = base.A


def negll_flat(params, y, S):
    gamma, delta = params
    e = y - gamma
    if len(S) == 0:
        return 0.5 * np.sum(e * e) + 0.5 * len(y) * math.log(2 * math.pi)
    kz = base.k0(delta * base.r_of(e, S))
    if not np.isfinite(kz).all() or np.any(kz <= 0):
        return 1e12
    z = base.Ez(delta, S)
    if not np.isfinite(z) or z <= 0:
        return 1e12
    return -( -0.5 * np.sum(e * e) - 0.5 * len(y) * math.log(2 * math.pi)
              + np.sum(np.log(kz)) - len(y) * math.log(z) )


def fit_flat(y, S):
    res = minimize(negll_flat, np.array([float(y.mean()), 0.0]), args=(y, S),
                   method="L-BFGS-B", bounds=[(-np.inf, np.inf), (-6.0, 6.0)],
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
            y = base.gen(n, rng, A)
            m = max(10, min(int(round(n ** (2 / 3))), max(10, n // 2)))
            yp = y[:m]
            ep = yp - yp.mean()
            Bp = base.bmat(ep)
            chat = (ep[:, None] * Bp).mean(axis=0)
            Jhat = float(np.mean(ep * ep))
            kap = np.abs(chat) / max(Jhat, 1e-12)
            lam = (math.log(base.K) ** 0.25) * n ** (-1 / 6)
            S = np.where(kap > lam)[0]
            y_main = y[m:]
            N = len(y_main)
            shift = base.true_mean(A, n)
            naive = float(y_main.mean()) - shift
            naive_se = float(y_main.std(ddof=1) / math.sqrt(N))
            pair = np.array([0, 1])
            map_c, _ = base.fit(y_main, pair)
            mle_c, _ = fit_flat(y_main, pair)
            map_se = base.se(y_main, map_c, pair)
            mle_se = base.se(y_main, mle_c, pair)
            if len(S) > 0:
                rmap, _ = base.fit(y_main, S)
                rmle, _ = fit_flat(y_main, S)
                rmap_se = base.se(y_main, rmap, S)
                rmle_se = base.se(y_main, rmle, S)
            else:
                rmap = rmle = naive
                rmap_se = rmle_se = naive_se
            rows.append(dict(n=n, rep=_, shift=shift, naive=naive,
                             naive_se=naive_se, map=map_c - shift,
                             map_se=map_se, mle=mle_c - shift, mle_se=mle_se,
                             rand_map=rmap - shift, rand_map_se=rmap_se,
                             rand_mle=rmle - shift, rand_mle_se=rmle_se,
                             nS=len(S), both=int(set(S.tolist()) == {0, 1}),
                             lam=lam))
        print(f"n={n} done ({time.time()-t0:.1f}s)", flush=True)
    with open(out / "results_p10_mle.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for n in NS:
        sub = [r for r in rows if r["n"] == n]
        summary[str(n)] = {}
        for key in ("map", "mle", "rand_map", "rand_mle"):
            b = float(np.mean([r[key] for r in sub]))
            cov = float(np.mean([abs(r[key]) <= 1.96 * r[key + "_se"] for r in sub]))
            summary[str(n)][key] = dict(bias=b, coverage=cov)
        summary[str(n)]["both_selected"] = float(np.mean([r["both"] for r in sub]))
    (out / "summary_p10_mle.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
