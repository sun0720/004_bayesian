#!/usr/bin/env python3
"""Non-degenerate P10 arm for h001-gptsol (review 012, P1-8).

Two dictionary directions with comparable nonzero sensitivities:
  b1 = alpha*u + beta*w1,  b2 = alpha*u + beta*(-0.5*w1 + 0.866*w2),
with u = sin(1.5 eps)/N1 (E[eps u] = 0.6925), w1,w2 orthonormal cosines
orthogonal to u.  alpha = 0.4/0.6925, beta = sqrt(1-alpha^2), so
E[eps b1] = E[eps b2] = 0.4 and the pair is orthonormal.  Then
J_{1,2} = 1 - 0.4^2 - 0.4^2 = 0.68 > c = 0.5, so the pair is admissible.

The perturbation is a local tilt along b1 + b2 (a1 = a2 = 0.75).  The pilot
selects both directions with probability tending to one, and the main-sample
augmentation uses a scalar delta along r_S = 0.4 b1 + 0.4 b2.  This is a
scalar-delta special case of the vector augmentation; it exercises the
non-degenerate selection that the one-direction arms cannot.
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

GX = np.linspace(-10.0, 10.0, 60001)
GW = np.exp(-0.5 * GX * GX) / math.sqrt(2.0 * math.pi)
GW = GW / np.trapezoid(GW, GX)
DX = GX[1] - GX[0]


def ip(f, g):
    return float(np.sum(GW * f * g) * DX)


def unit(f):
    return f / math.sqrt(ip(f, f))


U = unit(np.sin(1.5 * GX))
W1 = unit(np.cos(1.5 * GX) - ip(np.ones_like(GX), np.cos(1.5 * GX)) * np.ones_like(GX))
_w2 = np.cos(2.5 * GX) - ip(np.ones_like(GX), np.cos(2.5 * GX)) * np.ones_like(GX)
_w2 = _w2 - ip(U, _w2) * U - ip(W1, _w2) * W1
W2 = unit(_w2)
ALPHA = 0.4 / ip(GX, U)
BETA = math.sqrt(1.0 - ALPHA ** 2)
B1 = ALPHA * U + BETA * W1
B2 = ALPHA * U + BETA * (-0.5 * W1 + math.sqrt(0.75) * W2)
C1 = ip(GX, B1)
C2 = ip(GX, B2)
C = np.array([C1, C2])
JS = 1.0 - C1 ** 2 - C2 ** 2
K = 2
R = 500
SEED = 20261024
NS = (300, 1000, 3000, 10000)
A = 0.75
PRIOR_SD = 0.1

_XQ, _WQ = np.polynomial.hermite.hermgauss(80)
_EQ = math.sqrt(2.0) * _XQ
_WQ = _WQ / math.sqrt(math.pi)


def k0(z):
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 2.0 / (1.0 + np.exp(-2.0 * z))


def bmat(e):
    e = np.asarray(e, dtype=float)
    u = np.sin(1.5 * e) / math.sqrt(ip(U, U))
    w1 = np.cos(1.5 * e)
    # use the analytic trigonometric forms; the grid is only for the constants
    return np.column_stack([ALPHA * u + BETA * w1,
                            ALPHA * u + BETA * (-0.5 * w1 + math.sqrt(0.75) * np.cos(2.5 * e))])


def r_of(e, S):
    e = np.asarray(e, dtype=float)
    out = np.zeros_like(e)
    B = bmat(e)
    for j in np.asarray(S).ravel():
        out = out + C[j] * B[:, j]
    return out


def Ez(delta, S):
    if len(S) == 0 or delta == 0.0:
        return 1.0
    return float((_WQ * k0(delta * r_of(_EQ, S))).sum())


def negll(params, y, S):
    gamma, delta = params
    e = y - gamma
    if len(S) == 0:
        return 0.5 * np.sum(e * e) + 0.5 * len(y) * math.log(2 * math.pi)
    kz = k0(delta * r_of(e, S))
    if not np.isfinite(kz).all() or np.any(kz <= 0):
        return 1e12
    z = Ez(delta, S)
    if not np.isfinite(z) or z <= 0:
        return 1e12
    return -( -0.5 * np.sum(e * e) - 0.5 * len(y) * math.log(2 * math.pi)
              + np.sum(np.log(kz)) - len(y) * math.log(z)
              - 0.5 * (delta / PRIOR_SD) ** 2 )


def fit(y, S):
    res = minimize(negll, np.array([float(y.mean()), 0.0]), args=(y, S),
                   method="L-BFGS-B", bounds=[(-np.inf, np.inf), (-6.0, 6.0)],
                   options={"maxiter": 150, "ftol": 1e-10})
    if not res.success or not np.isfinite(res.fun):
        return float(y.mean()), 0.0
    return float(res.x[0]), float(res.x[1])


def se(y, gamma, S):
    e = y - gamma
    B = bmat(e)
    tilde = e - B[:, S] @ C[S] if len(S) else e
    js = float(np.mean(tilde * tilde))
    return math.sqrt(1.0 / (js * len(y))) if js > 1e-12 else float(e.std(ddof=1) / math.sqrt(len(y)))


# DGP: phi(e) exp(t (b1+b2)) / Z
SX = np.linspace(-9.0, 9.0, 120001)
LOGW = -0.5 * SX * SX
_H = bmat(SX)
_cache = {}


def sampler(t):
    key = round(t, 12)
    if key not in _cache:
        w = np.exp(LOGW + t * (_H[:, 0] + _H[:, 1]) - (LOGW + t * (_H[:, 0] + _H[:, 1])).max())
        cdf = np.cumsum(w); cdf /= cdf[-1]
        mean = float(np.trapezoid(w * SX, SX) / np.trapezoid(w, SX))
        _cache[key] = (cdf, mean)
    return _cache[key]


def gen(n, rng, a):
    t = a / math.sqrt(n)
    cdf, _ = sampler(t)
    return np.interp(rng.random(n), cdf, SX)


def true_mean(a, n):
    return sampler(a / math.sqrt(n))[1]


def main():
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    rows = []
    t0 = time.time()
    for n in NS:
        rng = default_rng(SEED + n)
        for _ in range(R):
            y = gen(n, rng, A)
            m = max(10, min(int(round(n ** (2 / 3))), max(10, n // 2)))
            yp = y[:m]
            ep = yp - yp.mean()
            Bp = bmat(ep)
            chat = (ep[:, None] * Bp).mean(axis=0)
            Jhat = float(np.mean(ep * ep))
            kap = np.abs(chat) / max(Jhat, 1e-12)
            lam = (math.log(K) ** 0.25) * n ** (-1 / 6)
            S = np.where(kap > lam)[0]
            y_main = y[m:]
            N = len(y_main)
            naive = float(y_main.mean())
            naive_se = float(y_main.std(ddof=1) / math.sqrt(N))
            shift = true_mean(A, n)
            oracle = naive - shift
            S_pair = np.array([0, 1])
            fixed, _ = fit(y_main, S_pair)
            fixed_se = se(y_main, fixed, S_pair)
            if len(S) > 0:
                rand, _ = fit(y_main, S)
                rand_se = se(y_main, rand, S)
            else:
                rand, rand_se = naive, naive_se
            rows.append(dict(n=n, rep=_, shift=shift, naive=naive - shift,
                             naive_se=naive_se, fixed=fixed - shift,
                             fixed_se=fixed_se, rand=rand - shift,
                             rand_se=rand_se, oracle=oracle,
                             nS=len(S), both=int(set(S.tolist()) == {0, 1}),
                             lam=lam))
        print(f"n={n} done ({time.time()-t0:.1f}s)", flush=True)
    with open(out / "results_p10.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {}
    for n in NS:
        sub = [r for r in rows if r["n"] == n]
        summary[str(n)] = {}
        for key in ("fixed", "rand", "naive", "oracle"):
            b = np.mean([r[key] for r in sub])
            rmse = math.sqrt(np.mean([r[key] ** 2 for r in sub]))
            cov = np.mean([abs(r[key]) <= 1.96 * r[key + "_se"] for r in sub]) if key != "oracle" else np.nan
            summary[str(n)][key] = dict(bias=float(b), rmse=float(rmse), coverage=float(cov))
        summary[str(n)]["both_selected"] = float(np.mean([r["both"] for r in sub]))
        summary[str(n)]["mean_nS"] = float(np.mean([r["nS"] for r in sub]))
    (out / "summary_p10.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
