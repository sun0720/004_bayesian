#!/usr/bin/env python3
"""Non-vacuous nuisance-projection arm for h001-gptsol (review 012, R4/RD-4).

Model: Y = gamma + zeta * X + eps, X ~ N(mu,1), eps ~ N(0,1), mu = 0.5.
Target gamma, nuisance zeta.  I_{gamma zeta} = mu = 0.5, so the efficient
score for gamma is s = (1 - 0.4 X) eps, not the gamma-score eps: the
projection is non-vacuous.

Dictionary: B(x,y) = b(x) * eps with b(x) = x^2 - (3 + mu^2).  Then
E[X b(X)] = 0 (orthogonal to the zeta-score X eps) and E[b] = -2, so
c = E[s B] = E[b] = -2 is nonzero.  The perturbation is a local tilt
a B(x,y)/sqrt(n); A2 holds by construction.

Outputs: results_projection.csv, summary_projection.json.
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

MU = 0.5
J_EFF = 1.0 - MU ** 2 / (1.0 + MU ** 2)   # 0.8
C_TRUE = -2.0
OMEGA = None  # computed numerically below
A = 0.5
R = 200
SEED = 20261025
NS = (300, 1000)
PRIOR_SD = 0.1

_XQ, _WQ = np.polynomial.hermite.hermgauss(40)
_ZQ = math.sqrt(2.0) * _XQ
_WQ = _WQ / math.sqrt(math.pi)
_X = MU + _ZQ
_E = _ZQ
_XX, _EE = np.meshgrid(_X, _E, indexing="ij")
_WW = np.outer(_WQ, _WQ)


def bfun(x):
    return np.asarray(x, dtype=float) ** 2 - (3.0 + MU ** 2)


def k0(z):
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 2.0 / (1.0 + np.exp(-2.0 * z))


OMEGA = float(np.sum(_WW * bfun(_XX) ** 2 * _EE ** 2))
JS = J_EFF - C_TRUE ** 2 / OMEGA


def rfun(x, eps):
    return C_TRUE * bfun(x) * eps


def z_norm(delta):
    if delta == 0.0:
        return 1.0
    return float(np.sum(_WW * k0(delta * rfun(_XX, _EE))))


def negll(params, y, x):
    gamma, zeta, delta = params
    eps = y - gamma - zeta * x
    ll = -0.5 * np.sum(eps ** 2) - 0.5 * len(y) * math.log(2 * math.pi)
    if abs(delta) > 1e-8:
        kz = k0(delta * rfun(x, eps))
        if not np.isfinite(kz).all() or np.any(kz <= 0):
            return 1e12
        z = z_norm(delta)
        if not np.isfinite(z) or z <= 0:
            return 1e12
        ll += float(np.sum(np.log(kz))) - len(y) * math.log(z)
    ll -= 0.5 * (delta / PRIOR_SD) ** 2
    return -ll


def fit(y, x):
    zeta0 = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    gamma0 = float(y.mean() - zeta0 * x.mean())
    res = minimize(negll, np.array([gamma0, zeta0, 0.0]), args=(y, x),
                   method="L-BFGS-B",
                   bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (-6.0, 6.0)],
                   options={"maxiter": 200, "ftol": 1e-10})
    if not res.success or not np.isfinite(res.fun):
        return gamma0, zeta0, 0.0
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


def se(y, x, gamma, zeta):
    eps = y - gamma - zeta * x
    s = (1.0 - 0.4 * x) * eps
    B = bfun(x) * eps
    # augmented residual tilde = s - c/Omega * B
    tilde = s - C_TRUE / OMEGA * B
    js = float(np.mean(tilde * tilde))
    return math.sqrt(1.0 / (js * len(y))) if js > 1e-12 else float(np.std(eps) / math.sqrt(len(y)))


def sample(n, rng, a):
    # importance resampling from the tilted joint law
    t = a / math.sqrt(n)
    m = max(4 * n, 2000)
    x = MU + rng.standard_normal(m)
    e = rng.standard_normal(m)
    logw = t * bfun(x) * e
    w = np.exp(logw - logw.max())
    idx = rng.choice(m, size=n, replace=True, p=w / w.sum())
    return x[idx], e[idx]


def main():
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    rows = []
    t0 = time.time()
    for n in NS:
        rng = default_rng(SEED + n)
        for _ in range(R):
            x, eps = sample(n, rng, A)
            y = eps  # gamma0 = zeta0 = 0
            m = max(10, min(int(round(n ** (2 / 3))), max(10, n // 2)))
            xp, yp, ep = x[:m], y[:m], eps[:m]
            zeta_p = float(np.sum((xp - xp.mean()) * (yp - yp.mean())) / np.sum((xp - xp.mean()) ** 2))
            gamma_p = float(yp.mean() - zeta_p * xp.mean())
            ep = yp - gamma_p - zeta_p * xp
            c_hat = float(np.mean(bfun(xp) * ep ** 2))
            kap = abs(c_hat) / J_EFF
            lam = (math.log(1) ** 0.25) * n ** (-1 / 6) if False else n ** (-1 / 6)
            S = 1 if kap > lam else 0
            if S:
                g, z, d = fit(y[m:], x[m:])
                s = se(y[m:], x[m:], g, z)
            else:
                g = float(y[m:].mean()); z = 0.0; s = float(np.std(y[m:]) / math.sqrt(len(y[m:])))
            naive = float(y[m:].mean())
            naive_se = float(np.std(y[m:]) / math.sqrt(len(y[m:])))
            rows.append(dict(n=n, rep=_, gamma=g, se=s, naive=naive, naive_se=naive_se,
                             selected=S, kappa=kap, lam=lam, c_hat=c_hat))
        print(f"n={n} done ({time.time()-t0:.1f}s)", flush=True)
    with open(out / "results_projection.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {}
    for n in NS:
        sub = [r for r in rows if r["n"] == n]
        summary[str(n)] = dict(
            selection=float(np.mean([r["selected"] for r in sub])),
            mean_c=float(np.mean([r["c_hat"] for r in sub])),
            aug_bias=float(np.mean([r["gamma"] for r in sub])),
            aug_coverage=float(np.mean([abs(r["gamma"]) <= 1.96 * r["se"] for r in sub])),
            naive_bias=float(np.mean([r["naive"] for r in sub])),
            naive_coverage=float(np.mean([abs(r["naive"]) <= 1.96 * r["naive_se"] for r in sub])),
        )
    (out / "summary_projection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
